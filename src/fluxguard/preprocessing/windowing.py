"""CWRU signal conditioning and deterministic window generation."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from fluxguard.config import SignalConfig


def condition_signal(values: np.ndarray, config: SignalConfig) -> np.ndarray:
    """Remove DC/drift and apply a zero-phase high-pass Butterworth filter."""

    samples = np.asarray(values, dtype=np.float64).squeeze()
    if samples.ndim != 1 or samples.size < 3:
        raise ValueError("signal must be a one-dimensional array with at least 3 samples")
    detrended = signal.detrend(samples, type="linear")
    numerator, denominator = signal.butter(
        config.highpass_order,
        config.highpass_cutoff_hz,
        btype="highpass",
        fs=config.sampling_rate_hz,
    )
    pad_length = 3 * max(len(numerator), len(denominator))
    if samples.size <= pad_length:
        raise ValueError("signal is too short for the configured Butterworth filter")
    return signal.filtfilt(numerator, denominator, detrended)


@dataclass(frozen=True)
class Window:
    """A conditioned signal window and source metadata."""

    values: np.ndarray
    source_id: str
    start: int
    end: int
    label: str | None = None


def make_windows(
    values: np.ndarray,
    source_id: str,
    config: SignalConfig,
    label: str | None = None,
    condition: bool = True,
) -> list[Window]:
    """Create stable, non-padded windows from one source recording."""

    samples = condition_signal(values, config) if condition else np.asarray(values, dtype=float)
    if samples.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if samples.size < config.window_size:
        return []
    return [
        Window(
            values=samples[start : start + config.window_size].copy(),
            source_id=source_id,
            start=start,
            end=start + config.window_size,
            label=label,
        )
        for start in range(0, samples.size - config.window_size + 1, config.stride)
    ]
