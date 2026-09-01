"""Safe, explicit checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import SRCNN


def save_checkpoint(
    path: str | Path,
    *,
    model: SRCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scale: int,
    metrics: dict[str, float],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "model_config": model.configuration(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scale": scale,
            "metrics": metrics,
        },
        destination,
    )


def load_model(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[SRCNN, dict[str, Any]]:
    """Load a trusted project checkpoint and return an evaluation-mode model."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint does not contain a model state")
    configuration = checkpoint.get("model_config", {})
    hidden = configuration.get("hidden_channels", (64, 32))
    model = SRCNN(
        in_channels=int(configuration.get("in_channels", 3)),
        hidden_channels=(int(hidden[0]), int(hidden[1])),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint

