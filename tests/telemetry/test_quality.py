import numpy as np

from fluxguard.telemetry.quality import SensorStatus, check_signal_quality


def test_sensor_quality_healthy_signal() -> None:
    signal = np.sin(np.linspace(0, 10, 1024)) + np.random.normal(0, 0.05, 1024)
    report = check_signal_quality(signal)
    assert report.is_valid is True
    assert report.status == SensorStatus.HEALTHY
    assert report.variance > 0.1


def test_sensor_quality_disconnected_flatline() -> None:
    signal = np.zeros(1024)
    report = check_signal_quality(signal)
    assert report.is_valid is False
    assert report.status == SensorStatus.DISCONNECTED


def test_sensor_quality_stuck_at_constant() -> None:
    signal = np.full(1024, 3.3)
    report = check_signal_quality(signal)
    assert report.is_valid is False
    assert report.status == SensorStatus.STUCK_AT


def test_sensor_quality_saturation_clipping() -> None:
    signal = np.clip(np.sin(np.linspace(0, 50, 1024)) * 5.0, -1.0, 1.0)
    report = check_signal_quality(signal, saturation_limit=1.0)
    assert report.status == SensorStatus.SATURATED
    assert report.saturation_ratio > 0.05


def test_sensor_quality_nan_rejection() -> None:
    signal = np.array([1.0, 2.0, np.nan, 4.0])
    report = check_signal_quality(signal)
    assert report.is_valid is False
    assert report.status == SensorStatus.INVALID
