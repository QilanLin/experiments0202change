from __future__ import annotations

from typing import Any


class NoAcceleratorAvailableError(RuntimeError):
    """Raised when auto device selection cannot find CUDA or MPS."""


def _mps_is_available(torch_mod: Any) -> bool:
    backends = getattr(torch_mod, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False

    is_available = getattr(mps_backend, "is_available", None)
    if not callable(is_available):
        return False

    try:
        return bool(is_available())
    except Exception:
        return False


def select_torch_device(
    explicit_device: str | None = None,
    *,
    torch_mod: Any | None = None,
) -> str:
    """Select a torch device with a stable priority order and no CPU fallback."""
    if explicit_device:
        return explicit_device

    if torch_mod is None:
        import torch as torch_mod

    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        try:
            if cuda.is_available():
                return "cuda"
        except Exception:
            pass

    if _mps_is_available(torch_mod):
        return "mps"

    raise NoAcceleratorAvailableError(
        "Auto device selection found neither CUDA nor MPS. "
        "CPU fallback is disabled."
    )


def select_timesfm_backend(device: str) -> str:
    """Map a torch device string to TimesFM's backend selector."""
    device_str = str(device)
    if device_str.startswith(("cuda", "mps")):
        return "gpu"
    if device_str.startswith("cpu"):
        return "cpu"
    raise ValueError(f"Unsupported TimesFM device: {device}")
