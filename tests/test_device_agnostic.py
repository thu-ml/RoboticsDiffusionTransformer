"""Device-agnostic behaviour of the XPU / CUDA / CPU plumbing.

Everything except the last class runs anywhere (no accelerator, no checkpoint, no
robot stack): backend detection, the dispatch behind ``synchronize`` /
``empty_cache`` / ``manual_seed``, the device the inference wrappers pick, the
``--device`` flag of the deployment script and the cache release in the training
sampler. The last class builds a tiny random-weight RDT from config and compares
CPU against XPU, and is skipped when no XPU is present.

Run with ``python -m pytest tests``.
"""

import importlib
import sys
import types
import unittest
from unittest import mock

import torch

from models import device as dev


def _patch_availability(*, xpu: bool, cuda: bool):
    """Make ``is_available`` report exactly the given backends."""

    def fake_is_available(name: str) -> bool:
        if name == "xpu":
            return xpu
        if name == "cuda":
            return cuda
        return name == "cpu"

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.available_device_types(), ["xpu", "cuda", "cpu"])

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")
            self.assertEqual(dev.resolve_device(None), torch.device("cuda"))

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device_type(), "cpu")
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_default_device_carries_index(self):
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.default_device(1), torch.device("xpu:1"))
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device(1), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(
                dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1")
            )

    def test_device_type_from_tensor_string_and_device(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.device("cuda")), "cuda")
        self.assertEqual(dev.device_type(torch.zeros(1)), "cpu")

    def test_device_count_follows_backend(self):
        with _patch_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "device_count", return_value=2):
                self.assertEqual(dev.device_count(), 2)
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "device_count", return_value=8):
                self.assertEqual(dev.device_count(), 8)
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.device_count(), 0)


class DeviceDispatchTest(unittest.TestCase):
    """Calls that used to be ``torch.cuda.*`` reach the right backend only."""

    def test_synchronize_and_empty_cache_use_the_device_backend(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(torch.xpu, "synchronize") as xs,
                mock.patch.object(torch.cuda, "synchronize") as cs,
                mock.patch.object(torch.xpu, "empty_cache") as xe,
                mock.patch.object(torch.cuda, "empty_cache") as ce,
            ):
                dev.synchronize("xpu:0")
                dev.empty_cache(torch.device("xpu:0"))
                xs.assert_called_once()
                xe.assert_called_once()
                cs.assert_not_called()
                ce.assert_not_called()
                dev.synchronize(torch.zeros(1).device)
                dev.empty_cache("cpu")
                xs.assert_called_once()
                xe.assert_called_once()
                dev.synchronize("cuda:0")
                dev.empty_cache("cuda:0")
                cs.assert_called_once()
                ce.assert_called_once()

    def test_nothing_is_called_on_a_host_without_accelerators(self):
        with _patch_availability(xpu=False, cuda=False):
            with (
                mock.patch.object(torch.cuda, "synchronize") as cs,
                mock.patch.object(torch.cuda, "empty_cache") as ce,
                mock.patch.object(torch.cuda, "manual_seed") as cm,
            ):
                dev.synchronize()
                dev.empty_cache()
                dev.manual_seed(0)
                cs.assert_not_called()
                ce.assert_not_called()
                cm.assert_not_called()

    def test_manual_seed_seeds_every_available_accelerator(self):
        with _patch_availability(xpu=True, cuda=False):
            with (
                mock.patch.object(torch.xpu, "manual_seed") as xm,
                mock.patch.object(torch.cuda, "manual_seed") as cm,
            ):
                dev.manual_seed(7)
                xm.assert_called_once_with(7)
                cm.assert_not_called()
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "manual_seed") as cm:
                dev.manual_seed(7)
                cm.assert_called_once_with(7)


class InferenceWrapperDeviceTest(unittest.TestCase):
    """``RoboticDiffusionTransformerModel`` no longer hard-codes ``cuda``."""

    def _build(self, module_name, **kwargs):
        module = importlib.import_module(module_name)
        cls = module.RoboticDiffusionTransformerModel
        patches = [
            mock.patch.object(
                cls, "get_vision_encoder", return_value=(None, mock.MagicMock())
            ),
            mock.patch.object(cls, "get_policy", return_value=mock.MagicMock()),
            mock.patch.object(
                cls, "get_text_encoder", return_value=(None, mock.MagicMock())
            ),
            mock.patch.object(cls, "reset"),
            # Record where the normalisation statistics go instead of moving them,
            # so a fake host does not need a real accelerator.
            mock.patch.object(module, "torch"),
        ]
        mocks = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        self.tensor_to = mocks[-1].tensor.return_value.to
        return cls({}, **kwargs)

    def test_device_is_detected_when_omitted(self):
        for module_name in ("scripts.agilex_model", "scripts.maniskill_model"):
            with self.subTest(module=module_name):
                with _patch_availability(xpu=True, cuda=True):
                    self.assertEqual(
                        self._build(module_name).device, torch.device("xpu")
                    )
                with _patch_availability(xpu=False, cuda=True):
                    self.assertEqual(
                        self._build(module_name).device, torch.device("cuda")
                    )
                with _patch_availability(xpu=False, cuda=False):
                    self.assertEqual(
                        self._build(module_name).device, torch.device("cpu")
                    )

    def test_explicit_device_is_kept(self):
        for module_name in ("scripts.agilex_model", "scripts.maniskill_model"):
            with self.subTest(module=module_name):
                with _patch_availability(xpu=True, cuda=True):
                    self.assertEqual(
                        self._build(module_name, device="cpu").device,
                        torch.device("cpu"),
                    )

    def test_normalisation_statistics_follow_the_resolved_device(self):
        with _patch_availability(xpu=True, cuda=False):
            model = self._build("scripts.maniskill_model")
        self.assertEqual(self.tensor_to.call_count, 4)
        for call in self.tensor_to.call_args_list:
            self.assertEqual(call.args[0], model.device)
            self.assertEqual(model.device, torch.device("xpu"))

    def test_encode_instruction_defaults_to_the_model_device(self):
        model = self._build("scripts.maniskill_model", device="cpu")
        tokens = mock.MagicMock()
        model.text_tokenizer = mock.MagicMock(return_value={"input_ids": tokens})
        model.text_model = mock.MagicMock()
        model.encode_instruction("pick it up")
        tokens.to.assert_called_once_with(torch.device("cpu"))


class DeploymentScriptTest(unittest.TestCase):
    """``scripts.agilex_inference`` needs ROS and OpenCV, which are stubbed out here."""

    def setUp(self):
        stubs = {
            name: mock.MagicMock()
            for name in (
                "rospy",
                "cv2",
                "cv_bridge",
                "geometry_msgs",
                "geometry_msgs.msg",
                "nav_msgs",
                "nav_msgs.msg",
                "sensor_msgs",
                "sensor_msgs.msg",
                "std_msgs",
                "std_msgs.msg",
            )
        }
        patcher = mock.patch.dict(sys.modules, stubs)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("scripts.agilex_inference", None)
        self.addCleanup(sys.modules.pop, "scripts.agilex_inference", None)
        self.module = importlib.import_module("scripts.agilex_inference")

    def _parse(self, *extra):
        argv = [
            "agilex_inference",
            "--pretrained_model_name_or_path",
            "ckpt",
            "--lang_embeddings_path",
            "e.pt",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return self.module.get_arguments()

    def test_device_flag_defaults_to_detection(self):
        self.assertIsNone(self._parse().device)
        self.assertEqual(self._parse("--device", "xpu:0").device, "xpu:0")

    def test_policy_receives_the_device_flag(self):
        args = self._parse("--device", "xpu:1")
        with (
            mock.patch.object(self.module, "create_model") as create_model,
            mock.patch("builtins.open", mock.mock_open(read_data="common: {}\n")),
        ):
            self.module.make_policy(args)
        self.assertEqual(create_model.call_args.kwargs["device"], "xpu:1")

    def test_qpos_is_not_forced_onto_cuda(self):
        with open(self.module.__file__) as handle:
            code_lines = [line for line in handle if not line.lstrip().startswith("#")]
        self.assertFalse([line for line in code_lines if ".cuda()" in line])


class SamplerCacheReleaseTest(unittest.TestCase):
    def test_log_sample_res_releases_the_cache_of_the_prediction_device(self):
        from train import sample

        batch = {
            "data_indices": [0],
            "ctrl_freqs": torch.tensor([25]),
            "state_norm": torch.ones(1, 4),
            "images": torch.zeros(1, 1, 3, 2, 2),
            "states": torch.zeros(1, 1, 4),
            "actions": torch.zeros(1, 2, 4),
            "state_elem_mask": torch.ones(1, 4),
            "lang_attn_mask": torch.ones(1, 3, dtype=torch.bool),
            "lang_embeds": torch.zeros(1, 3, 5),
        }
        rdt = mock.MagicMock()
        rdt.predict_action.return_value = torch.zeros(1, 2, 4)
        vision = mock.MagicMock(return_value=torch.zeros(1, 7), hidden_size=7)
        accelerator = mock.MagicMock()
        accelerator.gather_for_metrics.side_effect = lambda tensors: tensors
        accelerator.gather.side_effect = lambda tensor: tensor
        accelerator.is_main_process = True
        args = types.SimpleNamespace(num_sample_batches=1, precomp_lang_embed=True)
        with mock.patch.object(sample, "empty_cache") as empty_cache:
            sample.log_sample_res(
                None,
                vision,
                rdt,
                args,
                accelerator,
                torch.float32,
                {0: "toy"},
                [batch],
                mock.MagicMock(),
            )
        empty_cache.assert_called_once()
        self.assertEqual(empty_cache.call_args.args[0], torch.device("cpu"))


def _tiny_runner(dtype):
    from models.rdt_runner import RDTRunner

    config = {
        "rdt": {"hidden_size": 64, "depth": 2, "num_heads": 4},
        "lang_adaptor": "mlp2x_gelu",
        "img_adaptor": "mlp2x_gelu",
        "state_adaptor": "mlp3x_gelu",
        "noise_scheduler": {
            "num_train_timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
            "clip_sample": False,
            "num_inference_timesteps": 5,
        },
    }
    runner = RDTRunner(
        action_dim=16,
        pred_horizon=8,
        config=config,
        lang_token_dim=24,
        img_token_dim=20,
        state_token_dim=16,
        max_lang_cond_len=10,
        img_cond_len=12,
        img_pos_embed_config=[("image", (2, 3, -2))],
        lang_pos_embed_config=[("lang", -10)],
        dtype=dtype,
    )
    # The final layer is zero-initialised, which would make every output zero and the
    # comparison vacuous, so give every parameter a seeded random value.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in runner.parameters():
            param.copy_(
                (torch.randn(param.shape, generator=generator) * 0.1).to(param.dtype)
            )
    return runner.eval()


def _tiny_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    lang_mask = torch.zeros(1, 10, dtype=torch.bool)
    lang_mask[:, :7] = True
    return {
        "lang_tokens": torch.randn(1, 10, 24, generator=generator),
        "lang_attn_mask": lang_mask,
        "img_tokens": torch.randn(1, 12, 20, generator=generator),
        "state_tokens": torch.randn(1, 1, 16, generator=generator),
        "action_mask": torch.ones(1, 1, 16),
        "ctrl_freqs": torch.tensor([25]),
    }, torch.randn(1, 8, 16, generator=generator)


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an Intel XPU"
)
class XpuParityTest(unittest.TestCase):
    """Tiny random-weight RDT: CPU against XPU, same weights, inputs and noise."""

    def _predict(self, runner, device, dtype, inputs, noise):
        runner = runner.to(device, dtype=dtype)
        real_randn = torch.randn

        def fixed_noise(*args, **kwargs):
            if kwargs.get("size") == tuple(noise.shape):
                return noise.to(kwargs["device"], dtype=kwargs["dtype"])
            return real_randn(*args, **kwargs)

        moved = {
            key: value.to(device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device)
            for key, value in inputs.items()
        }
        with torch.no_grad(), mock.patch("torch.randn", fixed_noise):
            return runner.predict_action(**moved).float().cpu()

    def test_predict_action_matches_cpu_in_float32(self):
        runner = _tiny_runner(torch.float32)
        inputs, noise = _tiny_inputs(1)
        cpu = self._predict(
            _tiny_runner(torch.float32), "cpu", torch.float32, inputs, noise
        )
        xpu = self._predict(runner, "xpu", torch.float32, inputs, noise)
        self.assertGreater(cpu.abs().max().item(), 0.0)
        torch.testing.assert_close(xpu, cpu, atol=1e-3, rtol=1e-3)

    def test_predict_action_matches_cpu_in_bfloat16(self):
        inputs, noise = _tiny_inputs(2)
        cpu = self._predict(
            _tiny_runner(torch.bfloat16), "cpu", torch.bfloat16, inputs, noise
        )
        xpu = self._predict(
            _tiny_runner(torch.bfloat16), "xpu", torch.bfloat16, inputs, noise
        )
        self.assertTrue(torch.isfinite(xpu).all())
        torch.testing.assert_close(xpu, cpu, atol=5e-2, rtol=5e-2)

    def test_a_training_step_runs_on_xpu(self):
        runner = _tiny_runner(torch.float32).to("xpu").train()
        inputs, _ = _tiny_inputs(3)
        inputs = {key: value.to("xpu") for key, value in inputs.items()}
        action_gt = torch.randn(1, 8, 16, device="xpu")
        before = [param.detach().clone() for param in runner.parameters()]
        optimizer = torch.optim.AdamW(runner.parameters(), lr=1e-3)
        loss = runner(
            inputs["lang_tokens"],
            inputs["lang_attn_mask"],
            inputs["img_tokens"],
            inputs["state_tokens"],
            action_gt,
            inputs["action_mask"],
            inputs["ctrl_freqs"],
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, runner.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
