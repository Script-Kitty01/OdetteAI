"""Model metrics and serializable evaluation summaries."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("targets and predictions must be non-empty with equal shape")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "per_class": [
            {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for p, r, f, s in zip(precision, recall, f1, support)
        ],
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("targets and predictions must be non-empty with equal shape")
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
    }


def anomaly_metrics(targets: np.ndarray, errors: np.ndarray, threshold: float) -> dict[str, float]:
    targets = np.asarray(targets, dtype=int)
    errors = np.asarray(errors, dtype=float)
    predictions = errors > threshold
    result = classification_metrics(targets, predictions.astype(int))
    result["false_positive_rate"] = float(np.mean(predictions[targets == 0])) if np.any(targets == 0) else 0.0
    result["anomaly_recall"] = float(np.mean(predictions[targets == 1])) if np.any(targets == 1) else 0.0
    if np.unique(targets).size == 2:
        result["roc_auc"] = float(roc_auc_score(targets, errors))
    return result
