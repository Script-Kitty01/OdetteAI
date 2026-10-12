"""LSTM baseline for NASA C-MAPSS remaining useful life regression."""

import torch
from torch import nn


class CMapssRULRegressor(nn.Module):
    """Sequence-to-one LSTM regressor."""

    def __init__(self, sensor_count: int, hidden_size: int = 64, layers: int = 1) -> None:
        super().__init__()
        if sensor_count < 1 or hidden_size < 1 or layers < 1:
            raise ValueError("sensor_count, hidden_size, and layers must be positive")
        self.sensor_count = sensor_count
        self.lstm = nn.LSTM(sensor_count, hidden_size, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2 or 1), nn.ReLU(), nn.Linear(hidden_size // 2 or 1, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.sensor_count:
            raise ValueError("RUL input must have shape (batch, sequence, sensor_count)")
        output, _ = self.lstm(values)
        return self.head(output[:, -1]).squeeze(-1)
