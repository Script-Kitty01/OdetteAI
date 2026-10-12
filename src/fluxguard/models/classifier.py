"""Compact 1D CNN baseline for CWRU fault classification."""

import torch
from torch import nn


class CWRUClassifier(nn.Module):
    """A small classifier suitable for CPU smoke training and inference."""

    def __init__(self, class_count: int, input_length: int = 2048) -> None:
        super().__init__()
        if class_count < 2 or input_length < 16:
            raise ValueError("class_count must be >= 2 and input_length must be >= 16")
        self.class_count = class_count
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("classifier input must have shape (batch, 1, length) or (batch, length)")
        return self.head(self.features(values).squeeze(-1))
