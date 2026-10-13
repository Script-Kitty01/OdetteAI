"""Sensor health and data quality monitoring for industrial telemetry."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SensorStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    SATURATED = "saturated"
    STUCK_AT = "stuck_at"
    INVALID = "invalid"


@dataclass(frozen=True)
class SensorHealthReport:
    """Diagnostic outcome of sensor data quality validation."""

    status: SensorStatus
    is_valid: bool
    reasons: tuple[str, ...]
    mean_val: float
    variance: float
    min_val: float
    max_val: float
    saturation_ratio: float
    stuck_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "reasons": list(self.reasons),
            "mean": self.mean_val,
            "variance": self.variance,
            "min": self.min_val,
            "max": self.max_val,
            "saturation_ratio": self.saturation_ratio,
            "stuck_ratio": self.stuck_ratio,
        }


def check_signal_quality(
    signal: np.ndarray,
    saturation_limit: float | None = None,
    min_variance: float = 1e-7,
    max_stuck_samples: int = 32,
) -> SensorHealthReport:
    """Validate 1D telemetry against common sensor failure modes.

    Detects:
    - Non-finite (NaN, inf) samples.
    - Disconnected or flatline sensor (near-zero variance).
    - Sensor stuck at a constant non-zero value.
    - Sensor ADC rail saturation or clipping.
    """
    arr = np.asarray(signal, dtype=np.float64).squeeze()

    if arr.ndim != 1 or arr.size < 2:
        return SensorHealthReport(
            status=SensorStatus.INVALID,
            is_valid=False,
            reasons=("Signal must be a 1D array with at least 2 samples",),
            mean_val=0.0,
            variance=0.0,
            min_val=0.0,
            max_val=0.0,
            saturation_ratio=0.0,
            stuck_ratio=1.0,
        )

    if not np.isfinite(arr).all():
        return SensorHealthReport(
            status=SensorStatus.INVALID,
            is_valid=False,
            reasons=("Signal contains non-finite (NaN or Inf) values",),
            mean_val=0.0,
            variance=0.0,
            min_val=0.0,
            max_val=0.0,
            saturation_ratio=0.0,
            stuck_ratio=0.0,
        )

    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    reasons: list[str] = []
    status = SensorStatus.HEALTHY

    # 1. Disconnection / Zero Flatline Check
    if np.isclose(var_val, 0.0, atol=min_variance):
        if np.isclose(mean_val, 0.0, atol=1e-5):
            status = SensorStatus.DISCONNECTED
            reasons.append("Sensor disconnected (zero flatline detected)")
        else:
            status = SensorStatus.STUCK_AT
            reasons.append(f"Sensor stuck at constant value {mean_val:.4f}")

    # 2. Consecutive identical samples (stuck-at sequence)
    diffs = np.diff(arr)
    zero_diffs = np.isclose(diffs, 0.0, atol=1e-9)
    max_consecutive_stuck = 0
    current_consecutive = 0
    for zd in zero_diffs:
        if zd:
            current_consecutive += 1
            max_consecutive_stuck = max(max_consecutive_stuck, current_consecutive)
        else:
            current_consecutive = 0

    stuck_ratio = float(max_consecutive_stuck / arr.size)
    if max_consecutive_stuck >= max_stuck_samples and status == SensorStatus.HEALTHY:
        status = SensorStatus.DEGRADED
        reasons.append(f"Sensor exhibits stuck-at pattern ({max_consecutive_stuck} consecutive identical samples)")

    # 3. Saturation / Clipping Check
    saturation_ratio = 0.0
    if saturation_limit is not None and saturation_limit > 0:
        saturated_samples = np.sum(np.abs(arr) >= (saturation_limit * 0.999))
        saturation_ratio = float(saturated_samples / arr.size)
        if saturation_ratio > 0.02:
            status = SensorStatus.SATURATED
            reasons.append(f"Sensor signal saturated/clipped ({saturation_ratio * 100:.1f}% samples at limit)")

    is_valid = status in (SensorStatus.HEALTHY, SensorStatus.DEGRADED)
    return SensorHealthReport(
        status=status,
        is_valid=is_valid,
        reasons=tuple(reasons) if reasons else ("Sensor signals healthy and nominal",),
        mean_val=mean_val,
        variance=var_val,
        min_val=min_val,
        max_val=max_val,
        saturation_ratio=saturation_ratio,
        stuck_ratio=stuck_ratio,
    )
