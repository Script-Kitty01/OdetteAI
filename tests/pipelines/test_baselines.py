import numpy as np
import pytest

from fluxguard.pipelines.baselines import (
    feature_logistic_baseline,
    linear_sensor_baseline,
    majority_class_baseline,
    mean_regressor_baseline,
    rms_energy_anomaly_scores,
)


def test_majority_class_baseline_predicts_most_frequent_label() -> None:
    predictions = majority_class_baseline(np.array([1, 1, 2, 3]), 5)
    assert predictions.tolist() == [1, 1, 1, 1, 1]


def test_majority_class_baseline_rejects_empty_labels() -> None:
    with pytest.raises(ValueError):
        majority_class_baseline(np.array([]), 3)


def test_mean_regressor_baseline_predicts_training_mean() -> None:
    predictions = mean_regressor_baseline(np.array([2.0, 4.0]), 3)
    assert predictions.tolist() == [3.0, 3.0, 3.0]


def test_rms_energy_scores_order_windows_by_magnitude() -> None:
    windows = np.array([[0.0, 0.0], [1.0, 1.0]])
    scores = rms_energy_anomaly_scores(windows)
    assert scores[0] < scores[1]


def test_feature_logistic_baseline_separates_offset_classes() -> None:
    rng = np.random.default_rng(0)
    negative = rng.normal(-2.0, 0.2, size=(40, 64))
    positive = rng.normal(2.0, 0.2, size=(40, 64))
    train = np.vstack([negative[:30], positive[:30]])
    labels = np.array([0] * 30 + [1] * 30)
    evaluation = np.vstack([negative[30:], positive[30:]])
    predictions = feature_logistic_baseline(train, labels, evaluation)
    assert np.mean(predictions == np.array([0] * 10 + [1] * 10)) >= 0.9


def test_linear_sensor_baseline_recovers_linear_targets() -> None:
    features = np.arange(10, dtype=float).reshape(-1, 1)
    targets = 3.0 * features[:, 0] + 1.0
    predictions = linear_sensor_baseline(features, targets, np.array([[20.0]]))
    assert predictions[0] == pytest.approx(61.0)
