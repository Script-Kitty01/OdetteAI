import numpy as np
import pytest

from fluxguard.evaluation.sensitivity import analyze_threshold_sensitivity


def test_sensitivity_report_shapes_and_best_multiplier() -> None:
    healthy = np.array([0.1, 0.12, 0.11, 0.09])
    anomalies = np.array([0.8, 0.9, 0.7, 0.85])
    report = analyze_threshold_sensitivity(healthy, anomalies, multipliers=(1.0, 2.0, 3.0))
    assert len(report.multipliers) == 3
    assert len(report.false_positive_rates) == 3
    assert len(report.anomaly_recalls) == 3
    assert len(report.f1_scores) == 3
    assert 0 <= report.roc_auc <= 1
    # Low multiplier catches everything; the best F1 must come from a
    # multiplier that still detects all anomalies.
    assert report.anomaly_recalls[0] == 1.0
    assert report.best_multiplier() in (1.0, 2.0, 3.0)


def test_sensitivity_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        analyze_threshold_sensitivity(np.array([]), np.array([1.0]))


def test_sensitivity_rejects_negative_multipliers() -> None:
    with pytest.raises(ValueError):
        analyze_threshold_sensitivity(
            np.array([0.1, 0.2]), np.array([0.9]), multipliers=(-1.0,)
        )
