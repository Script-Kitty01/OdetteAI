"""Short-time Fourier transform and compact time-frequency representations."""

from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import stft


@dataclass(frozen=True)
class STFTFeatures:
    """One-sided STFT magnitude and its coordinate axes."""

    frequencies_hz: np.ndarray
    times_s: np.ndarray
    magnitude: np.ndarray
    sampling_rate_hz: float
    n_fft: int
    hop_length: int


def extract_stft(
    signal: np.ndarray,
    sampling_rate_hz: float,
    n_fft: int = 256,
    hop_length: int = 128,
    output_size: tuple[int, int] | None = None,
) -> STFTFeatures:
    """Extract a finite, one-sided STFT magnitude from one vibration window."""

    values = np.asarray(signal, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size < 2:
        raise ValueError("signal must be a one-dimensional array with at least two samples")
    if not np.isfinite(values).all():
        raise ValueError("signal must contain only finite values")
    if sampling_rate_hz <= 0 or n_fft < 2 or hop_length < 1:
        raise ValueError("sampling rate, n_fft, and hop_length must be positive")
    segment_length = min(n_fft, values.size)
    overlap = min(segment_length - 1, segment_length - hop_length)
    frequencies, times, transformed = stft(
        values,
        fs=sampling_rate_hz,
        nperseg=segment_length,
        noverlap=max(0, overlap),
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(transformed).astype(np.float32)
    if output_size is not None:
        rows, columns = output_size
        if rows < 1 or columns < 1:
            raise ValueError("output_size dimensions must be positive")
        magnitude = _resize_nearest(magnitude, output_size)
    return STFTFeatures(
        frequencies.astype(np.float32),
        times.astype(np.float32),
        magnitude,
        float(sampling_rate_hz),
        n_fft,
        hop_length,
    )


def _resize_nearest(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_indices = np.linspace(0, values.shape[0] - 1, shape[0]).round().astype(int)
    column_indices = np.linspace(0, values.shape[1] - 1, shape[1]).round().astype(int)
    return values[np.ix_(row_indices, column_indices)].astype(np.float32)


def extract_cwt(
    signal: np.ndarray,
    sampling_rate_hz: float,
    scales: np.ndarray | None = None,
    wavelet: str = "morl",
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a finite CWT magnitude matrix for compatible scalogram experiments."""

    values = np.asarray(signal, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional array with at least two samples")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    selected_scales = np.asarray(scales if scales is not None else np.arange(1, 65), dtype=float)
    if selected_scales.ndim != 1 or selected_scales.size == 0 or np.any(selected_scales <= 0):
        raise ValueError("scales must be a non-empty vector of positive values")
    coefficients, _ = pywt.cwt(values, selected_scales, wavelet, sampling_period=1 / sampling_rate_hz)
    magnitude = np.abs(coefficients).astype(np.float32)
    return _resize_nearest(magnitude, output_size) if output_size is not None else magnitude
