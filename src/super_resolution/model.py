"""SRCNN fidelity baseline used in controlled experiments."""

from __future__ import annotations

import torch
from torch import nn


class SRCNN(nn.Module):
    """Three-layer Super-Resolution Convolutional Neural Network.

    The network refines an image that has already been enlarged with bicubic
    interpolation. Spatial dimensions are preserved by symmetric padding.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        if in_channels <= 0 or any(channel <= 0 for channel in hidden_channels):
            raise ValueError("channel counts must be positive")
        first_hidden, second_hidden = hidden_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, first_hidden, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(first_hidden, second_hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(second_hidden, in_channels, kernel_size=5, padding=2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("input must have shape (batch, channels, height, width)")
        if image.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, received {image.shape[1]}")
        return self.features(image)

    def configuration(self) -> dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "hidden_channels": list(self.hidden_channels),
        }
