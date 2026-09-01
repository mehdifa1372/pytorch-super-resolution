"""Training and evaluation loops."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .data import SuperResolutionDataset
from .device import resolve_device
from .metrics import peak_signal_to_noise_ratio, structural_similarity
from .model import SRCNN


@dataclass(frozen=True)
class TrainingConfig:
    train_dir: Path
    validation_dir: Path
    train_manifest: Path | None = None
    validation_manifest: Path | None = None
    output_dir: Path = Path("artifacts")
    scale: int = 2
    patch_size: int = 96
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-4
    workers: int = 0
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if self.scale < 2:
            raise ValueError("scale must be at least 2")
        if self.patch_size <= 0 or self.patch_size % self.scale:
            raise ValueError("patch_size must be positive and divisible by scale")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    examples = 0
    criterion = nn.MSELoss(reduction="mean")
    with torch.inference_mode():
        for bicubic, target in loader:
            bicubic = bicubic.to(device)
            target = target.to(device)
            prediction = model(bicubic).clamp(0, 1)
            batch_size = target.shape[0]
            total_loss += float(criterion(prediction, target)) * batch_size
            total_psnr += float(peak_signal_to_noise_ratio(prediction, target)) * batch_size
            total_ssim += float(structural_similarity(prediction, target)) * batch_size
            examples += batch_size
    if examples == 0:
        raise ValueError("evaluation loader produced no examples")
    return {
        "loss": total_loss / examples,
        "psnr": total_psnr / examples,
        "ssim": total_ssim / examples,
    }


def train_model(config: TrainingConfig) -> list[dict[str, float]]:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    training_dataset = SuperResolutionDataset(
        config.train_dir,
        scale=config.scale,
        patch_size=config.patch_size,
        training=True,
        manifest=config.train_manifest,
    )
    validation_dataset = SuperResolutionDataset(
        config.validation_dir,
        scale=config.scale,
        patch_size=config.patch_size,
        training=False,
        manifest=config.validation_manifest,
    )
    generator = torch.Generator().manual_seed(config.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )

    model = SRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_psnr = float("-inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        progress = tqdm(training_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for bicubic, target in progress:
            bicubic = bicubic.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(bicubic)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            batch_size = target.shape[0]
            running_loss += float(loss.detach()) * batch_size
            examples += batch_size
            progress.set_postfix(loss=f"{running_loss / examples:.6f}")

        validation = evaluate_model(model, validation_loader, device)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": running_loss / examples,
            "validation_loss": validation["loss"],
            "validation_psnr": validation["psnr"],
            "validation_ssim": validation["ssim"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

        save_checkpoint(
            config.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scale=config.scale,
            metrics=epoch_metrics,
        )
        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            save_checkpoint(
                config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scale=config.scale,
                metrics=epoch_metrics,
            )

    serializable_config = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    (config.output_dir / "run.json").write_text(
        json.dumps({"config": serializable_config, "history": history}, indent=2),
        encoding="utf-8",
    )
    return history
