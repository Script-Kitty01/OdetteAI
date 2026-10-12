"""Training helper for the NASA C-MAPSS LSTM RUL baseline."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fluxguard.models.rul import CMapssRULRegressor
from fluxguard.pipelines.common import set_seed


@dataclass(frozen=True)
class RULHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def train_rul(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    validation_sequences: np.ndarray,
    validation_targets: np.ndarray,
    *,
    hidden_size: int = 64,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[CMapssRULRegressor, RULHistory]:
    train_x, train_y = _inputs(train_sequences, train_targets)
    validation_x, validation_y = _inputs(validation_sequences, validation_targets)
    if train_x.shape[2] != validation_x.shape[2]:
        raise ValueError("train and validation sequences must have the same sensor count")
    set_seed(seed)
    model = CMapssRULRegressor(train_x.shape[2], hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
            count += len(batch_x)
        train_losses.append(total / count)
        model.eval()
        with torch.no_grad():
            validation_losses.append(float(loss_fn(model(validation_x.to(device)), validation_y.to(device)).item()))
    return model, RULHistory(tuple(train_losses), tuple(validation_losses))


def predict_rul(model: CMapssRULRegressor, sequences: np.ndarray, device: str = "cpu") -> np.ndarray:
    values = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    model = model.to(device).eval()
    with torch.no_grad():
        return model(values.to(device)).cpu().numpy()


def _inputs(sequences: np.ndarray, targets: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.float32)
    if values.ndim != 3 or labels.ndim != 1 or values.shape[0] != labels.shape[0]:
        raise ValueError("sequences must be 3D and targets must match the first dimension")
    if values.shape[0] == 0 or not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("training data must be non-empty and finite")
    return torch.from_numpy(values), torch.from_numpy(labels)
