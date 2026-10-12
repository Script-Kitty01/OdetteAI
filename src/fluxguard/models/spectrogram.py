"""Compact 2D CNN for STFT/CWT spectrogram representations."""

import torch
from torch import nn


class SpectrogramCNN(nn.Module):
    """Small conv net over single-channel time-frequency images."""

    def __init__(self, class_count: int, height: int = 32, width: int = 32) -> None:
        super().__init__()
        if class_count < 2 or height < 8 or width < 8:
            raise ValueError("class_count must be >= 2 and spatial dims must be >= 8")
        self.class_count = class_count
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if values.ndim == 3:
                values = values.unsqueeze(1)
            if values.ndim != 4 or values.shape[1] != 1:
                raise ValueError(
                    "spectrogram input must have shape (batch, 1, height, width) or (batch, height, width)"
                )
        elif values.ndim == 3:
            values = values.unsqueeze(1)
        features = self.features(values).flatten(1)
        return self.head(features)