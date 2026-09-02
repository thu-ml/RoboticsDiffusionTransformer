import copy

import pytest
import torch
from accelerate import Accelerator

from models.ema_model import EMAModel


def make_model(value):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def set_weight(model, value):
    with torch.no_grad():
        model.weight.fill_(value)


def test_ema_steps_once_per_completed_optimizer_update():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4)
    model = make_model(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = EMAModel(copy.deepcopy(accelerator.unwrap_model(model)), power=0.75)
    ema_micro_steps = []

    for micro_step in range(8):
        with accelerator.accumulate(model):
            loss = model(torch.ones(1, 1)).sum()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        optimizer_step_succeeded = ema.step_if_optimizer_updated(
            accelerator.unwrap_model(model),
            sync_gradients=accelerator.sync_gradients,
            optimizer_step_was_skipped=accelerator.optimizer_step_was_skipped,
        )
        if optimizer_step_succeeded:
            ema_micro_steps.append(micro_step + 1)

    assert ema_micro_steps == [4, 8]
    assert ema.optimization_step == 2


def test_ema_does_not_advance_when_optimizer_step_is_skipped():
    model = make_model(1.0)
    ema = EMAModel(make_model(0.0))

    updated = ema.step_if_optimizer_updated(
        model,
        sync_gradients=True,
        optimizer_step_was_skipped=True,
    )

    assert not updated
    assert ema.optimization_step == 0
    assert ema.averaged_model.weight.item() == 0.0


def test_restored_optimization_step_matches_uninterrupted_ema(tmp_path):
    model = make_model(0.0)
    uninterrupted = EMAModel(copy.deepcopy(model), power=0.75)
    for value in range(1, 7):
        set_weight(model, value)
        uninterrupted.step(model)

    resumed_model = make_model(0.0)
    before_checkpoint = EMAModel(copy.deepcopy(resumed_model), power=0.75)
    for value in range(1, 4):
        set_weight(resumed_model, value)
        before_checkpoint.step(resumed_model)

    state_path = tmp_path / "ema_state.pt"
    torch.save(before_checkpoint.schedule_state_dict(), state_path)

    resumed = EMAModel(copy.deepcopy(before_checkpoint.averaged_model), power=0.75)
    resumed.load_schedule_state_dict(torch.load(state_path, map_location="cpu"))
    for value in range(4, 7):
        set_weight(resumed_model, value)
        resumed.step(resumed_model)

    assert resumed.optimization_step == uninterrupted.optimization_step
    assert resumed.decay == uninterrupted.decay
    assert torch.equal(
        resumed.averaged_model.weight, uninterrupted.averaged_model.weight
    )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_optimization_step_must_be_a_non_negative_integer(value):
    ema = EMAModel(make_model(0.0))

    with pytest.raises((TypeError, ValueError)):
        ema.set_optimization_step(value)
