"""Statistical features for conditioned vibration windows."""

import numpy as np
from scipy.stats import kurtosis, skew


class FeatureExtractionError(ValueError):
    """Raised when a signal cannot produce valid statistical features."""


def extract_time_domain_features(values: np.ndarray) -> dict[str, float]:
    """Return finite time-domain features for one non-zero 1D signal.

    Kurtosis uses SciPy's Fisher convention (normal distribution == 0). For a
    non-zero constant signal, skewness is defined as 0 and Fisher kurtosis as
    -3 because the standardized moments are otherwise undefined.
    """

    signal = np.asarray(values, dtype=np.float64).squeeze()
    if signal.ndim != 1 or signal.size < 2:
        raise FeatureExtractionError("values must be a 1D array with at least 2 samples")
    if not np.isfinite(signal).all():
        raise FeatureExtractionError("values must contain only finite samples")

    mean = float(np.mean(signal))
    standard_deviation = float(np.std(signal))
    absolute_peak = float(np.max(np.abs(signal)))
    peak_to_peak = float(np.ptp(signal))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    if np.isclose(rms, 0.0):
        raise FeatureExtractionError("RMS is zero; ratio features are undefined")

    if np.isclose(standard_deviation, 0.0):
        skewness = 0.0
        fisher_kurtosis = -3.0
    else:
        skewness = float(skew(signal, bias=False))
        fisher_kurtosis = float(kurtosis(signal, fisher=True, bias=False))

    return {
        "mean": mean,
        "std": standard_deviation,
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "absolute_peak": absolute_peak,
        "skewness": skewness,
        "kurtosis": fisher_kurtosis,
        "crest_factor": absolute_peak / rms,
        "impulse_factor": absolute_peak / float(np.mean(np.abs(signal))),
        "form_factor": rms / float(np.mean(np.abs(signal))),
    }


def feature_matrix(values: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Extract ordered features from a ``(windows, samples)`` matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise FeatureExtractionError("values must be a 2D windows-by-samples array")
    extracted = [extract_time_domain_features(row) for row in matrix]
    names = list(extracted[0]) if extracted else []
    return names, np.asarray([[row[name] for name in names] for row in extracted])
