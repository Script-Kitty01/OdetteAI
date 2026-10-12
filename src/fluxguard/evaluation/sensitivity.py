"""Threshold sensitivity analysis for anomaly detectors."""

from dataclasses import dataclass

import numpy as np

from fluxguard.models.autoencoder import AnomalyThreshold


@dataclass(frozen=True)
class ThresholdSensitivityReport:
    """Sweep of threshold multipliers with resulting operating-point metrics."""

    multipliers: tuple[float, ...]
    false_positive_rates: tuple[float, ...]
    anomaly_recalls: tuple[float, ...]
    f1_scores: tuple[float, ...]
    roc_auc: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "multipliers": list(self.multipliers),
            "false_positive_rates": list(self.false_positive_rates),
            "anomaly_recalls": list(self.anomaly_recalls),
            "f1_scores": list(self.f1_scores),
            "roc_auc": self.roc_auc,
        }

    def best_multiplier(self) -> float:
        """Return the multiplier with the highest F1 (ties break toward lower FPR)."""

        if not self.f1_scores:
            raise ValueError("sensitivity report is empty")
        best_index = max(
            range(len(self.f1_scores)),
            key=lambda index: (self.f1_scores[index], -self.false_positive_rates[index]),
        )
        return self.multipliers[best_index]


def analyze_threshold_sensitivity(
    healthy_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
) -> ThresholdSensitivityReport:
    """Evaluate precision-oriented operating points across threshold multipliers.

    The threshold at each multiplier is ``mean(healthy) + k * std(healthy)`` so
    the sweep mirrors how :class:`AnomalyThreshold` is calibrated in practice.
    """

    healthy = np.asarray(healthy_errors, dtype=float)
    anomalies = np.asarray(anomaly_errors, dtype=float)
    if healthy.ndim != 1 or anomalies.ndim != 1 or healthy.size == 0 or anomalies.size == 0:
        raise ValueError("healthy_errors and anomaly_errors must be non-empty vectors")
    if not np.isfinite(healthy).all() or not np.isfinite(anomalies).all():
        raise ValueError("error vectors must be finite")
    if not multipliers or any(multiplier < 0 for multiplier in multipliers):
        raise ValueError("multipliers must be non-empty and non-negative")

    mean = float(healthy.mean())
    std = float(healthy.std())
    false_positive_rates: list[float] = []
    anomaly_recalls: list[float] = []
    f1_scores: list[float] = []
    for multiplier in multipliers:
        threshold = AnomalyThreshold(mean, std, float(multiplier)).threshold
        false_positive_rates.append(float(np.mean(healthy > threshold)))
        anomaly_recalls.append(float(np.mean(anomalies > threshold)))
        tp = float(np.sum(anomalies > threshold))
        fp = float(np.sum(healthy > threshold))
        fn = float(np.sum(anomalies <= threshold))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_scores.append(f1)

    roc_auc: float | None = None
    labels = np.concatenate([np.zeros(healthy.size), np.ones(anomalies.size)])
    scores = np.concatenate([healthy, anomalies])
    if np.unique(labels).size == 2:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(labels, scores))

    return ThresholdSensitivityReport(
        multipliers=tuple(float(multiplier) for multiplier in multipliers),
        false_positive_rates=tuple(false_positive_rates),
        anomaly_recalls=tuple(anomaly_recalls),
        f1_scores=tuple(f1_scores),
        roc_auc=roc_auc,
    )