"""Squeeze-and-excitation temporal convolutional network classifier."""

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    """Channel-wise squeeze-and-excitation gating."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = self.gate(self.pool(values).squeeze(-1)).unsqueeze(-1)
        return values * weights


class TemporalBlock(nn.Module):
    """Causal dilated convolution with a residual connection."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(values, (self.padding, 0))
        out = self.dropout(self.activation(self.norm(self.conv(padded))))
        return self.activation(out + self.residual(values))


class SETCNClassifier(nn.Module):
    """Compact SE-TCN baseline for raw CWRU vibration windows."""

    def __init__(
        self,
        class_count: int,
        input_length: int = 2048,
        channels: tuple[int, ...] = (32, 64, 96),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if class_count < 2 or input_length < 16:
            raise ValueError("class_count must be >= 2 and input_length must be >= 16")
        if not channels or kernel_size < 2 or not 0 <= dropout < 1:
            raise ValueError("channels must be non-empty, kernel_size >= 2, dropout in [0, 1)")
        self.class_count = class_count
        blocks: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, 2**index, dropout))
            blocks.append(SqueezeExcitation(out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels[-1], class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if values.ndim == 2:
                values = values.unsqueeze(1)
            if values.ndim != 3 or values.shape[1] != 1:
                raise ValueError("SE-TCN input must have shape (batch, 1, length) or (batch, length)")
        elif values.ndim == 2:
            values = values.unsqueeze(1)
        features = self.pool(self.blocks(values)).squeeze(-1)
        return self.head(features)