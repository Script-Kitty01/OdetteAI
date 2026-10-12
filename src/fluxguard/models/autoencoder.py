"""Compact 1D convolutional autoencoder for healthy CWRU windows."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class CWRUAutoencoder(nn.Module):
    """CPU-friendly convolutional autoencoder preserving input length."""

    def __init__(self, input_length: int = 2048, latent_channels: int = 16) -> None:
        super().__init__()
        if input_length < 8 or latent_channels < 1:
            raise ValueError("input_length must be >= 8 and latent_channels must be positive")
        self.input_length = input_length
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, latent_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if not torch.jit.is_tracing() and (values.ndim != 3 or values.shape[1] != 1):
            raise ValueError("autoencoder input must have shape (batch, 1, length) or (batch, length)")
        reconstructed = self.decoder(self.encoder(values))
        return reconstructed[..., : values.shape[-1]]

    @torch.no_grad()
    def reconstruction_errors(self, values: torch.Tensor) -> torch.Tensor:
        reconstructed = self(values)
        return torch.mean((reconstructed - (values.unsqueeze(1) if values.ndim == 2 else values)) ** 2, dim=(1, 2))


@dataclass(frozen=True)
class AnomalyThreshold:
    """Threshold calibrated from healthy validation reconstruction errors."""

    mean_error: float
    std_error: float
    multiplier: float = 3.0

    @property
    def threshold(self) -> float:
        return self.mean_error + self.multiplier * self.std_error

    def predict(self, errors: np.ndarray) -> np.ndarray:
        return np.asarray(errors, dtype=float) > self.threshold


def calibrate_threshold(errors: np.ndarray, multiplier: float = 3.0) -> AnomalyThreshold:
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("errors must be a non-empty finite vector")
    if multiplier < 0:
        raise ValueError("multiplier cannot be negative")
    return AnomalyThreshold(float(values.mean()), float(values.std()), float(multiplier))
