"""NASA C-MAPSS parsing, RUL labeling, and training-only sensor pruning."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class CMapssDataError(ValueError):
    """Raised when C-MAPSS data violates its documented schema."""


SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3", *SENSOR_COLUMNS]


@dataclass(frozen=True)
class SensorSelection:
    """Training-fitted sensor selection metadata."""

    selected: tuple[str, ...]
    dropped: tuple[str, ...]
    variances: dict[str, float]
    threshold: float


def read_cmapss(path: Path) -> pd.DataFrame:
    """Read one C-MAPSS train/test file with strict column validation."""

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except (OSError, ValueError) as exc:
        raise CMapssDataError(f"Could not read C-MAPSS file {path}: {exc}") from exc
    if frame.shape[1] != len(COLUMNS):
        raise CMapssDataError(
            f"{path} has {frame.shape[1]} columns; expected {len(COLUMNS)}"
        )
    frame.columns = COLUMNS
    if frame[["engine_id", "cycle"]].isna().any().any():
        raise CMapssDataError(f"{path} contains missing engine IDs or cycles")
    for engine_id, group in frame.groupby("engine_id", sort=False):
        if not group["cycle"].is_monotonic_increasing:
            raise CMapssDataError(f"Engine {engine_id} cycles are not monotonic in {path}")
    return frame


def add_training_rul(frame: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Add remaining-cycle RUL labels to a training frame."""

    result = frame.copy()
    final_cycles = result.groupby("engine_id")["cycle"].transform("max")
    rul = final_cycles - result["cycle"]
    result["rul"] = rul.clip(upper=cap) if cap is not None else rul
    return result


def read_test_rul(path: Path) -> np.ndarray:
    """Read one C-MAPSS test RUL file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    if values.empty or values.isna().any():
        raise CMapssDataError(f"{path} does not contain valid RUL values")
    return values.to_numpy(dtype=np.float64)


def fit_sensor_selection(
    training: pd.DataFrame,
    variance_threshold: float = 1e-12,
) -> SensorSelection:
    """Fit variance pruning on training rows only."""

    if variance_threshold < 0:
        raise ValueError("variance_threshold cannot be negative")
    missing = sorted(set(SENSOR_COLUMNS) - set(training.columns))
    if missing:
        raise CMapssDataError(f"Missing sensor columns: {missing}")
    variances = training[SENSOR_COLUMNS].var(axis=0, ddof=0).astype(float).to_dict()
    selected = tuple(name for name in SENSOR_COLUMNS if variances[name] > variance_threshold)
    dropped = tuple(name for name in SENSOR_COLUMNS if name not in selected)
    if not selected:
        raise CMapssDataError("Variance pruning removed every sensor")
    return SensorSelection(selected, dropped, variances, variance_threshold)


def apply_sensor_selection(frame: pd.DataFrame, selection: SensorSelection) -> pd.DataFrame:
    """Return metadata columns plus the fitted sensor selection."""

    metadata = [column for column in ("engine_id", "cycle", "rul") if column in frame]
    return frame[metadata + list(selection.selected)].copy()


def make_engine_sequences(
    frame: pd.DataFrame,
    sensor_columns: Iterable[str],
    sequence_length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build engine-local sequences and final-step targets without crossing engines."""

    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    missing = sorted(set(sensor_columns) - set(frame.columns))
    if missing:
        raise CMapssDataError(f"Missing sequence sensor columns: {missing}")
    if "rul" not in frame:
        raise CMapssDataError("frame must contain a rul column")
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    engines: list[int] = []
    for engine_id, group in frame.groupby("engine_id", sort=True):
        ordered = group.sort_values("cycle")
        values = ordered[list(sensor_columns)].to_numpy(dtype=np.float32)
        labels = ordered["rul"].to_numpy(dtype=np.float32)
        for start in range(0, max(0, len(ordered) - sequence_length + 1), stride):
            end = start + sequence_length
            sequences.append(values[start:end])
            targets.append(float(labels[end - 1]))
            engines.append(int(engine_id))
    if not sequences:
        return (
            np.empty((0, sequence_length, len(tuple(sensor_columns))), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(sequences), np.asarray(targets), np.asarray(engines)
