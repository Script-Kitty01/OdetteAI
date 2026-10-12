"""Typed configuration for the FluxGuard data foundation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SignalConfig:
    """CWRU conditioning and window parameters."""

    sampling_rate_hz: float = 12_000.0
    highpass_cutoff_hz: float = 10.0
    highpass_order: int = 4
    window_size: int = 2_048
    overlap: float = 0.5

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if not 0 < self.highpass_cutoff_hz < self.sampling_rate_hz / 2:
            raise ValueError("highpass_cutoff_hz must be below the Nyquist frequency")
        if self.highpass_order < 1:
            raise ValueError("highpass_order must be at least 1")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")

    @property
    def stride(self) -> int:
        return max(1, round(self.window_size * (1 - self.overlap)))


@dataclass(frozen=True)
class CMapssConfig:
    """NASA C-MAPSS sequence and sensor-selection parameters."""

    variance_threshold: float = 1e-12
    sequence_length: int = 30
    sequence_stride: int = 1
    rul_cap: int | None = 125
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold cannot be negative")
        if self.sequence_length < 1 or self.sequence_stride < 1:
            raise ValueError("sequence_length and sequence_stride must be positive")
        if self.rul_cap is not None and self.rul_cap < 1:
            raise ValueError("rul_cap must be positive when provided")


@dataclass(frozen=True)
class DataConfig:
    """Repository data locations."""

    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
