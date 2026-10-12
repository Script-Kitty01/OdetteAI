import numpy as np
import pytest

from fluxguard.features.frequency_domain import (
    FrequencyFeatureError,
    band_energy,
    extract_fft_features,
)
from fluxguard.features.time_domain import FeatureExtractionError, extract_time_domain_features


def test_time_domain_features_are_finite_and_expected_for_sine() -> None:
    values = np.sin(np.linspace(0, 2 * np.pi, 1_001))

    features = extract_time_domain_features(values)

    assert all(np.isfinite(value) for value in features.values())
    assert features["rms"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert features["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)


def test_time_domain_rejects_zero_signal() -> None:
    with pytest.raises(FeatureExtractionError, match="RMS is zero"):
        extract_time_domain_features(np.zeros(32))


def test_fft_identifies_dominant_frequency_and_band_energy() -> None:
    sampling_rate = 1_000.0
    time = np.arange(1_000) / sampling_rate
    values = np.sin(2 * np.pi * 100 * time)

    features = extract_fft_features(values, sampling_rate)

    assert features.dominant_frequency_hz == pytest.approx(100.0)
    assert features.frequency_resolution_hz == pytest.approx(1.0)
    assert band_energy(features, 90, 110) > band_energy(features, 200, 300)


def test_fft_rejects_invalid_frequency_band() -> None:
    features = extract_fft_features(np.ones(32), 1_000.0)

    with pytest.raises(FrequencyFeatureError, match="band"):
        band_energy(features, 300, 600)
