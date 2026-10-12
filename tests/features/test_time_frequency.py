import numpy as np

from fluxguard.features.time_frequency import extract_cwt, extract_stft


def test_stft_returns_one_sided_finite_magnitude() -> None:
    signal = np.sin(2 * np.pi * 100 * np.arange(1024) / 1000)
    result = extract_stft(signal, 1000, n_fft=128, hop_length=64, output_size=(32, 16))
    assert result.magnitude.shape == (32, 16)
    assert result.frequencies_hz.shape == (65,)
    assert np.isfinite(result.magnitude).all()


def test_stft_rejects_non_finite_signal() -> None:
    with np.testing.assert_raises(ValueError):
        extract_stft(np.array([1.0, np.nan]), 1000)


def test_cwt_returns_compatible_scalogram() -> None:
    result = extract_cwt(np.ones(256), 1000, output_size=(16, 32))
    assert result.shape == (16, 32)
    assert np.isfinite(result).all()
