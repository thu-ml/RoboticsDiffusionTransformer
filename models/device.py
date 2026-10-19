"""Device-agnostic helpers for the XPU / CUDA / CPU backends.

Every place that used to call ``torch.cuda`` directly goes through this module, so
the inference, evaluation and training entry points share one code path across
accelerators.

Two rules this module encodes:

* A wheel built without CUDA (an Intel XPU or CPU-only install) raises
  ``AssertionError: Torch not compiled with CUDA enabled`` from calls such as
  ``torch.cuda.manual_seed`` or ``torch.cuda.empty_cache``. Callers use
  :func:`manual_seed` / :func:`empty_cache`, which only touch a backend that is
  available.
* Detection order is XPU, then CUDA, then CPU, and an explicit ``device=``
  argument always wins over detection. On a CUDA-only host the result is the same
  as before this module existed.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

__all__ = [
    "available_device_types",
    "default_device",
    "default_device_type",
    "device_count",
    "device_type",
    "empty_cache",
    "is_available",
    "manual_seed",
    "resolve_device",
    "synchronize",
]

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def available_device_types() -> list[str]:
    """Accelerator types on this host in detection order, ending with ``cpu``."""
    return [name for name in _ACCELERATORS if is_available(name)] + ["cpu"]


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    return available_device_types()[0]


def default_device(index: Optional[int] = None) -> torch.device:
    """Device of the best available backend, optionally with an explicit index."""
    name = default_device_type()
    if index is None or name == "cpu":
        return torch.device(name)
    return torch.device(name, int(index))


def resolve_device(device: DeviceLike) -> torch.device:
    """Resolve ``None`` / str / :class:`torch.device` to a concrete device.

    ``None`` auto-detects with :func:`default_device`. An explicit value is honoured
    verbatim, so an unsupported device still fails at the first kernel instead of
    being replaced by another backend.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return default_device()
    return torch.device(device)


def device_type(device: Any) -> str:
    """Backend type (``xpu``, ``cuda`` or ``cpu``) of a device, string or tensor."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def _resolved_type(device: DeviceLike) -> str:
    return device_type(device) if device is not None else default_device_type()


def device_count(backend: Optional[str] = None) -> int:
    """Number of visible devices of a backend; 0 when the backend is unavailable."""
    name = backend or default_device_type()
    if name == "xpu":
        return torch.xpu.device_count() if is_available("xpu") else 0
    if name == "cuda":
        return torch.cuda.device_count() if is_available("cuda") else 0
    return 0


def synchronize(device: DeviceLike = None) -> None:
    """Synchronize the accelerator of ``device``; no-op on CPU."""
    name = _resolved_type(device)
    if name == "xpu" and is_available("xpu"):
        torch.xpu.synchronize()
    elif name == "cuda" and is_available("cuda"):
        torch.cuda.synchronize()


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator blocks of ``device``; no-op on CPU."""
    name = _resolved_type(device)
    if name == "xpu" and is_available("xpu"):
        torch.xpu.empty_cache()
    elif name == "cuda" and is_available("cuda"):
        torch.cuda.empty_cache()


def manual_seed(seed: int) -> None:
    """Seed the RNG of every available accelerator, like ``torch.cuda.manual_seed``."""
    if is_available("xpu"):
        torch.xpu.manual_seed(seed)
    if is_available("cuda"):
        torch.cuda.manual_seed(seed)
