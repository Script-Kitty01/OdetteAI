"""Matplotlib report figures saved inside experiment run directories."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_loss: tuple[float, ...] | list[float],
    validation_loss: tuple[float, ...] | list[float],
    path: Path,
    title: str = "Training curves",
) -> None:
    """Save a train/validation loss line chart."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, len(train_loss) + 1), train_loss, label="train")
    axis.plot(range(1, len(validation_loss) + 1), validation_loss, label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], path: Path, title: str = "Confusion matrix"
) -> None:
    """Save a labelled confusion-matrix heatmap."""

    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(max(4, 0.6 * values.shape[1] + 2), 4.5))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(values.shape[1]), class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(values.shape[0]), class_names, fontsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center", fontsize=8)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_error_distributions(
    healthy_errors: np.ndarray, anomaly_errors: np.ndarray, path: Path, threshold: float
) -> None:
    """Save healthy-vs-anomaly reconstruction error histograms with the threshold line."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(healthy_errors, bins=40, alpha=0.6, label="healthy", color="tab:green")
    axis.hist(anomaly_errors, bins=40, alpha=0.6, label="fault", color="tab:red")
    axis.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4g}")
    axis.set_xlabel("reconstruction error")
    axis.set_ylabel("count")
    axis.set_title("Anomaly score distributions")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_threshold_sensitivity(
    multipliers: list[float] | tuple[float, ...],
    false_positive_rates: list[float] | tuple[float, ...],
    anomaly_recalls: list[float] | tuple[float, ...],
    path: Path,
) -> None:
    """Save FPR/recall curves across threshold multipliers."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(multipliers, false_positive_rates, marker="o", label="healthy FPR")
    axis.plot(multipliers, anomaly_recalls, marker="o", label="anomaly recall")
    axis.set_xlabel("threshold multiplier (mean + k·std)")
    axis.set_ylabel("rate")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Threshold sensitivity")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_prediction_scatter(
    targets: np.ndarray, predictions: np.ndarray, path: Path, title: str = "Predicted vs actual"
) -> None:
    """Save a predicted-vs-actual scatter with the identity line."""

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(targets, predictions, s=8, alpha=0.5)
    low = float(min(np.min(targets), np.min(predictions)))
    high = float(max(np.max(targets), np.max(predictions)))
    axis.plot([low, high], [low, high], "r--", linewidth=1)
    axis.set_xlabel("actual")
    axis.set_ylabel("predicted")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_rul_examples(
    cycles: list[list[float]], actual: list[list[float]], predicted: list[list[float]], path: Path
) -> None:
    """Save RUL trajectories for a few example engines."""

    figure, axis = plt.subplots(figsize=(7, 4))
    for index, (x, y_true, y_pred) in enumerate(zip(cycles, actual, predicted)):
        axis.plot(x, y_true, label=f"engine {index + 1} actual", color=f"C{index}")
        axis.plot(x, y_pred, linestyle="--", label=f"engine {index + 1} predicted", color=f"C{index}")
    axis.set_xlabel("cycle")
    axis.set_ylabel("RUL")
    axis.set_title("RUL trajectories")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_metric_bars(names: list[str], values: list[float], path: Path, title: str) -> None:
    """Save a simple bar chart comparing named scalar metrics."""

    figure, axis = plt.subplots(figsize=(max(5, 0.8 * len(names)), 4))
    axis.bar(names, values, color="tab:blue")
    axis.set_ylabel("score")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30, labelsize=8)
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
