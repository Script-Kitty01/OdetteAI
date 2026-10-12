"""Training helper for the CWRU compact 1D CNN classifier."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fluxguard.models.classifier import CWRUClassifier
from fluxguard.pipelines.common import set_seed


@dataclass(frozen=True)
class ClassifierHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def train_classifier(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    validation_windows: np.ndarray,
    validation_labels: np.ndarray,
    *,
    class_count: int,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[CWRUClassifier, ClassifierHistory]:
    train_x, train_y = _inputs(train_windows, train_labels)
    validation_x, validation_y = _inputs(validation_windows, validation_labels)
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train and validation windows must have the same length")
    set_seed(seed)
    model = CWRUClassifier(class_count, train_x.shape[1]).to(device)
    counts = torch.bincount(train_y, minlength=class_count).float()
    weights = torch.where(counts > 0, counts.sum() / (counts * class_count), torch.zeros_like(counts))
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
            validation_losses.append(float(loss_fn(model(validation_x.to(device)), validation_y.to(device)).item()))
    return model, ClassifierHistory(tuple(train_losses), tuple(validation_losses))


def predict_classifier(model: CWRUClassifier, windows: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    values = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    model = model.to(device).eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(values.to(device)), dim=1).cpu().numpy()
    return probabilities.argmax(axis=1), probabilities


def _inputs(windows: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.asarray(windows, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.ndim != 1 or values.shape[0] != targets.shape[0]:
        raise ValueError("windows must be 2D and labels must match the first dimension")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("training data must be non-empty and finite")
    return torch.from_numpy(values), torch.from_numpy(targets)
