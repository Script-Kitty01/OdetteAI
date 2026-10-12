"""Generic classification trainer for 1D windows and 2D spectrogram batches."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fluxguard.pipelines.common import set_seed


@dataclass(frozen=True)
class ClassificationHistory:
    """Per-epoch mean training and validation losses."""

    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def train_classifier_model(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    *,
    model_factory: Callable[[], nn.Module],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[nn.Module, ClassificationHistory]:
    """Train any classification network with class-weighted cross entropy."""

    train_x, train_y, validation_x, validation_y = _inputs(
        train_inputs, train_labels, validation_inputs, validation_labels
    )
    set_seed(seed)
    model = model_factory().to(device)
    counts = torch.bincount(train_y, minlength=int(train_y.max().item()) + 1).float()
    weights = torch.where(counts > 0, counts.sum() / (counts * counts.numel()), torch.zeros_like(counts))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            validation_losses.append(
                float(loss_fn(model(validation_x.to(device)), validation_y.to(device)).item())
            )
    return model, ClassificationHistory(tuple(train_losses), tuple(validation_losses))


def predict_classifier_model(
    model: nn.Module, inputs: np.ndarray, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class indices and softmax probabilities."""

    values = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    model = model.to(device).eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(values.to(device)), dim=1).cpu().numpy()
    return probabilities.argmax(axis=1), probabilities


def _inputs(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = np.asarray(train_inputs, dtype=np.float32)
    train_y = np.asarray(train_labels, dtype=np.int64)
    validation_x = np.asarray(validation_inputs, dtype=np.float32)
    validation_y = np.asarray(validation_labels, dtype=np.int64)
    if train_x.ndim not in (2, 3) or validation_x.ndim != train_x.ndim:
        raise ValueError("inputs must be 2D windows or 3D spectrograms with matching ranks")
    if train_x.shape[1:] != validation_x.shape[1:]:
        raise ValueError("train and validation inputs must share non-batch dimensions")
    if train_y.ndim != 1 or validation_y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if train_x.shape[0] != train_y.shape[0] or validation_x.shape[0] != validation_y.shape[0]:
        raise ValueError("labels must match the first input dimension")
    if (
        train_x.shape[0] == 0
        or validation_x.shape[0] == 0
        or not np.isfinite(train_x).all()
        or not np.isfinite(validation_x).all()
    ):
        raise ValueError("inputs must be non-empty and finite")
    return (
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(validation_x),
        torch.from_numpy(validation_y),
    )