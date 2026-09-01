from pathlib import Path

import torch

from super_resolution.checkpoints import load_model, save_checkpoint
from super_resolution.model import SRCNN


def test_checkpoint_round_trip(tmp_path: Path):
    model = SRCNN(hidden_channels=(8, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        scale=2,
        metrics={"validation_psnr": 20.0},
    )
    restored, metadata = load_model(path, device=torch.device("cpu"))
    assert restored.hidden_channels == (8, 4)
    assert metadata["epoch"] == 3
    assert metadata["scale"] == 2

