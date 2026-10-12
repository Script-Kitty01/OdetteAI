"""Training and inference helpers for the healthy-only CWRU autoencoder."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fluxguard.models.autoencoder import AnomalyThreshold, CWRUAutoencoder, calibrate_threshold
from fluxguard.pipelines.common import set_seed


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def train_autoencoder(
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[CWRUAutoencoder, AnomalyThreshold, TrainingHistory]:
    train = _window_tensor(train_windows)
    validation = _window_tensor(validation_windows)
    if train.shape[1] != validation.shape[1]:
        raise ValueError("train and validation windows must have the same length")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs and batch_size must be positive and learning_rate must be positive")
    set_seed(seed)
    model = CWRUAutoencoder(input_length=train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        running = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch).squeeze(1), batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch)
            count += len(batch)
        train_losses.append(running / count)
        model.eval()
        with torch.no_grad():
            errors = model.reconstruction_errors(validation.to(device))
        validation_losses.append(float(errors.mean().item()))
    model.eval()
    with torch.no_grad():
        validation_errors = model.reconstruction_errors(validation.to(device)).cpu().numpy()
    threshold = calibrate_threshold(validation_errors)
    return model, threshold, TrainingHistory(tuple(train_losses), tuple(validation_losses))


def score_autoencoder(model: CWRUAutoencoder, windows: np.ndarray, device: str = "cpu") -> np.ndarray:
    values = _window_tensor(windows).to(device)
    model = model.to(device).eval()
    with torch.no_grad():
        return model.reconstruction_errors(values).cpu().numpy()


def _window_tensor(windows: np.ndarray) -> torch.Tensor:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("windows must be a non-empty finite 2D array")
    return torch.from_numpy(values)
