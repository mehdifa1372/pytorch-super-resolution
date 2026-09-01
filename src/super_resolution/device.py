"""Runtime device selection."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve `auto`, `cpu`, `cuda`, or `mps` to an available torch device."""
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(normalized)

