"""Simple non-deep baselines recorded before model comparisons (plan gate 3)."""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from fluxguard.features.time_domain import feature_matrix


def majority_class_baseline(train_labels: np.ndarray, count: int) -> np.ndarray:
    """Predict the most frequent training label for every sample."""

    labels = np.asarray(train_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("train_labels must be a non-empty vector")
    if count < 1:
        raise ValueError("count must be positive")
    values, counts = np.unique(labels, return_counts=True)
    return np.full(count, values[int(np.argmax(counts))], dtype=labels.dtype)


def mean_regressor_baseline(train_targets: np.ndarray, count: int) -> np.ndarray:
    """Predict the training-target mean for every sample."""

    targets = np.asarray(train_targets, dtype=float)
    if targets.ndim != 1 or targets.size == 0 or not np.isfinite(targets).all():
        raise ValueError("train_targets must be a non-empty finite vector")
    if count < 1:
        raise ValueError("count must be positive")
    return np.full(count, float(targets.mean()))


def rms_energy_anomaly_scores(windows: np.ndarray) -> np.ndarray:
    """Per-window RMS energy used as a thresholdable anomaly score."""

    values = np.asarray(windows, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("windows must be a non-empty finite 2D array")
    return np.sqrt(np.mean(np.square(values), axis=1))


def feature_logistic_baseline(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    evaluation_windows: np.ndarray,
) -> np.ndarray:
    """Time-domain features plus logistic regression as a classical baseline."""

    names, train_features = feature_matrix(train_windows)
    evaluation_names, evaluation_features = feature_matrix(evaluation_windows)
    if names != evaluation_names:
        raise ValueError("train and evaluation feature schemas differ")
    labels = np.asarray(train_labels)
    if labels.ndim != 1 or labels.shape[0] != train_features.shape[0]:
        raise ValueError("train_labels must match the training windows")
    scaler = StandardScaler().fit(train_features)
    classifier = LogisticRegression(max_iter=2000).fit(
        scaler.transform(train_features), labels
    )
    return classifier.predict(scaler.transform(evaluation_features))


def linear_sensor_baseline(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    evaluation_features: np.ndarray,
) -> np.ndarray:
    """Last-step sensor linear regression as a classical RUL baseline."""

    train_x = np.asarray(train_features, dtype=float)
    evaluation_x = np.asarray(evaluation_features, dtype=float)
    targets = np.asarray(train_targets, dtype=float)
    if train_x.ndim != 2 or evaluation_x.ndim != 2 or train_x.shape[1] != evaluation_x.shape[1]:
        raise ValueError("train and evaluation features must be 2D with equal width")
    if train_x.shape[0] != targets.size or targets.size == 0:
        raise ValueError("train_targets must match the training feature rows")
    return LinearRegression().fit(train_x, targets).predict(evaluation_x)
