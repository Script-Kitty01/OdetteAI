import numpy as np
import pytest

from fluxguard.config import SignalConfig
from fluxguard.preprocessing.scaling import fit_training_scaler
from fluxguard.preprocessing.splits import group_split
from fluxguard.preprocessing.windowing import condition_signal, make_windows


def test_conditioning_removes_drift_and_produces_finite_signal() -> None:
    config = SignalConfig(sampling_rate_hz=1_000, highpass_cutoff_hz=10, window_size=100)
    time = np.arange(2_000) / config.sampling_rate_hz
    values = 4 + time + np.sin(2 * np.pi * 100 * time)

    conditioned = condition_signal(values, config)

    assert np.isfinite(conditioned).all()
    assert abs(float(np.mean(conditioned))) < 0.1


def test_windows_keep_source_recording_metadata() -> None:
    config = SignalConfig(sampling_rate_hz=1_000, highpass_cutoff_hz=10, window_size=100)
    windows = make_windows(np.ones(250), "recording-a", config, label="healthy", condition=False)

    assert len(windows) == 4
    assert {window.source_id for window in windows} == {"recording-a"}
    assert {window.label for window in windows} == {"healthy"}


def test_group_split_is_deterministic_and_disjoint() -> None:
    first = group_split(["a", "b", "c", "d", "e", "f"], seed=7)
    second = group_split(["f", "e", "d", "c", "b", "a"], seed=7)

    assert first == second
    assert not (set(first.train) & set(first.validation))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.validation) & set(first.test))


def test_scaler_rejects_non_finite_training_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_training_scaler(np.array([[1.0, np.nan]]), ("a", "b"))
