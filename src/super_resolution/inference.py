"""Checkpoint inference and directory evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .checkpoints import load_model
from .data import SuperResolutionDataset
from .device import resolve_device
from .images import pil_to_tensor, tensor_to_pil
from .training import evaluate_model


def upscale_image(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path,
    *,
    scale: int | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    """Upscale one low-resolution image using bicubic interpolation plus SRCNN."""
    device = resolve_device(device_name)
    model, checkpoint = load_model(checkpoint_path, device=device)
    checkpoint_scale = int(checkpoint.get("scale", 2))
    selected_scale = checkpoint_scale if scale is None else scale
    if selected_scale != checkpoint_scale:
        raise ValueError(
            f"requested x{selected_scale}, but the checkpoint was trained for x{checkpoint_scale}"
        )
    with Image.open(input_path) as source:
        rgb = source.convert("RGB")
        enlarged = rgb.resize(
            (rgb.width * selected_scale, rgb.height * selected_scale),
            Image.Resampling.BICUBIC,
        )
    input_tensor = pil_to_tensor(enlarged).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(input_tensor).clamp(0, 1)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(prediction).save(destination)
    return {
        "input": str(input_path),
        "output": str(destination),
        "scale": selected_scale,
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def evaluate_checkpoint(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    patch_size: int | None = None,
    batch_size: int = 4,
    workers: int = 0,
    device_name: str = "auto",
    manifest: str | Path | None = None,
) -> dict[str, object]:
    device = resolve_device(device_name)
    model, checkpoint = load_model(checkpoint_path, device=device)
    scale = int(checkpoint.get("scale", 2))
    dataset = SuperResolutionDataset(
        data_dir,
        scale=scale,
        patch_size=patch_size,
        training=False,
        manifest=manifest,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    metrics = evaluate_model(model, loader, device)
    bicubic_metrics = evaluate_model(torch.nn.Identity(), loader, device)
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "examples": len(dataset),
        "scale": scale,
        "device": str(device),
        "manifest": str(manifest) if manifest is not None else None,
        "srcnn": metrics,
        "bicubic": bicubic_metrics,
        "psnr_gain_db": metrics["psnr"] - bicubic_metrics["psnr"],
        "ssim_gain": metrics["ssim"] - bicubic_metrics["ssim"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
