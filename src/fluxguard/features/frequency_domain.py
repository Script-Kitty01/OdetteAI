"""FFT-based spectral features for vibration windows."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window


class FrequencyFeatureError(ValueError):
    """Raised when a signal cannot produce valid frequency features."""


@dataclass(frozen=True)
class FFTFeatures:
    """One-sided FFT output and its sampling metadata."""

    frequencies_hz: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray
    sampling_rate_hz: float
    frequency_resolution_hz: float
    dominant_frequency_hz: float


def extract_fft_features(
    values: np.ndarray,
    sampling_rate_hz: float,
    window: str = "hann",
) -> FFTFeatures:
    """Return a detrended, windowed, one-sided magnitude/power spectrum."""

    signal = np.asarray(values, dtype=np.float64).squeeze()
    if signal.ndim != 1 or signal.size < 2:
        raise FrequencyFeatureError("values must be a 1D array with at least 2 samples")
    if not np.isfinite(signal).all():
        raise FrequencyFeatureError("values must contain only finite samples")
    if sampling_rate_hz <= 0:
        raise FrequencyFeatureError("sampling_rate_hz must be positive")

    centered = signal - np.mean(signal)
    tapered = centered * get_window(window, signal.size, fftbins=True)
    spectrum = np.fft.rfft(tapered)
    magnitude = np.abs(spectrum) * (2.0 / signal.size)
    power = np.square(magnitude)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate_hz)
    dominant_index = int(np.argmax(magnitude[1:]) + 1) if magnitude.size > 1 else 0
    return FFTFeatures(
        frequencies_hz=frequencies,
        magnitude=magnitude,
        power=power,
        sampling_rate_hz=float(sampling_rate_hz),
        frequency_resolution_hz=float(sampling_rate_hz / signal.size),
        dominant_frequency_hz=float(frequencies[dominant_index]),
    )


def band_energy(features: FFTFeatures, low_hz: float, high_hz: float) -> float:
    """Sum spectral power within an inclusive frequency band."""

    if not 0 <= low_hz < high_hz <= features.sampling_rate_hz / 2:
        raise FrequencyFeatureError("band must satisfy 0 <= low_hz < high_hz <= Nyquist")
    mask = (features.frequencies_hz >= low_hz) & (features.frequencies_hz <= high_hz)
    return float(np.sum(features.power[mask]))
