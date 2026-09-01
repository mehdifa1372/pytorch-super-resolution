"""Image-quality metrics implemented with PyTorch."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Return mean batch PSNR in decibels."""
    _validate_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    error = (prediction - target).square().flatten(1).mean(dim=1)
    maximum = torch.tensor(data_range**2, dtype=error.dtype, device=error.device)
    score = 10.0 * torch.log10(maximum / error.clamp_min(torch.finfo(error.dtype).tiny))
    score = torch.where(error == 0, torch.full_like(score, torch.inf), score)
    return score.mean()


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Return mean SSIM using a local uniform window."""
    _validate_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    height, width = prediction.shape[-2:]
    window = min(window_size, height, width)
    if window % 2 == 0:
        window -= 1
    if window < 1:
        raise ValueError("images must have non-empty spatial dimensions")
    padding = window // 2

    mean_prediction = functional.avg_pool2d(prediction, window, 1, padding)
    mean_target = functional.avg_pool2d(target, window, 1, padding)
    variance_prediction = functional.avg_pool2d(prediction.square(), window, 1, padding)
    variance_prediction -= mean_prediction.square()
    variance_target = functional.avg_pool2d(target.square(), window, 1, padding)
    variance_target -= mean_target.square()
    covariance = functional.avg_pool2d(prediction * target, window, 1, padding)
    covariance -= mean_prediction * mean_target

    constant_one = (0.01 * data_range) ** 2
    constant_two = (0.03 * data_range) ** 2
    numerator = (2 * mean_prediction * mean_target + constant_one) * (
        2 * covariance + constant_two
    )
    denominator = (mean_prediction.square() + mean_target.square() + constant_one) * (
        variance_prediction + variance_target + constant_two
    )
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).mean()
