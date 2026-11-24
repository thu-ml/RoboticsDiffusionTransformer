import torch


def get_default_device(index=None):
    """Return the default device: CUDA if available, then Intel XPU, else CPU.

    Args:
        index: optional device index, used for CUDA and XPU.
    """
    if torch.cuda.is_available():
        device_type = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device_type = "xpu"
    else:
        return torch.device("cpu")
    if index is None:
        return torch.device(device_type)
    return torch.device(device_type, index)
