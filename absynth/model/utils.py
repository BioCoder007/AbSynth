import torch


def resolve_device(device: str) -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        device: "auto", "cpu", "cuda", or "mps". "auto" picks cuda > mps > cpu.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
