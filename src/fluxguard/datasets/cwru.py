"""CWRU MAT-file discovery, signal extraction, and metadata normalization."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


class CWRUDataError(ValueError):
    """Raised when a CWRU file cannot be interpreted safely."""


@dataclass(frozen=True)
class CWRURecord:
    """A source recording and its normalized metadata."""

    path: Path
    recording_id: str
    signal_name: str
    fault_type: str
    fault_size_in: float | None
    load: int | None
    sensor_location: str
    sampling_rate_hz: float


_SIGNAL_RE = re.compile(
    r"^(X\d+_DE_time|X\d+_FE_time|X\d+_BA_time)$", re.IGNORECASE
)
_FAULT_RE = re.compile(r"(?P<size>\d{3})", re.IGNORECASE)


def discover_mat_files(root: Path) -> list[Path]:
    """Return deterministic CWRU MAT files below *root*."""

    if not root.exists():
        raise FileNotFoundError(f"CWRU directory does not exist: {root}")
    files = sorted(root.rglob("*.mat"))
    if not files:
        raise CWRUDataError(f"No .mat files found under {root}")
    return files


def _metadata(path: Path, sampling_rate_hz: float) -> tuple[str, float | None, int | None]:
    stem = path.stem.lower()
    if stem in {"97", "98", "99", "100"}:
        return "healthy", None, None
    size_match = _FAULT_RE.search(stem)
    size = int(size_match.group("size")) / 1000 if size_match else None
    if "or" in stem:
        fault = "outer_race"
    elif "ir" in stem:
        fault = "inner_race"
    elif "ball" in stem or "b" in stem:
        fault = "ball"
    else:
        fault = "fault"
    load_match = re.search(r"(?:_|-)([0-3])$", stem)
    load = int(load_match.group(1)) if load_match else None
    return fault, size, load


def _find_signal(mat: dict[str, object]) -> tuple[str, np.ndarray]:
    candidates: list[tuple[str, np.ndarray]] = []
    for name, value in mat.items():
        if name.startswith("__") or not _SIGNAL_RE.match(name):
            continue
        array = np.asarray(value).squeeze()
        if array.ndim == 1 and array.size:
            candidates.append((name, array.astype(np.float64, copy=False)))
    if not candidates:
        raise CWRUDataError("No recognized vibration channel found")
    candidates.sort(key=lambda item: item[0])
    return candidates[0]


def load_record(path: Path, sampling_rate_hz: float = 12_000.0) -> tuple[CWRURecord, np.ndarray]:
    """Load one MAT file and return its normalized record and one signal."""

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        mat = loadmat(path)
    except (OSError, ValueError) as exc:
        raise CWRUDataError(f"Could not read MAT file {path}: {exc}") from exc
    signal_name, signal = _find_signal(mat)
    fault, size, load = _metadata(path, sampling_rate_hz)
    location = "drive_end" if "_DE_" in signal_name.upper() else (
        "fan_end" if "_FE_" in signal_name.upper() else "base"
    )
    record = CWRURecord(
        path=path,
        recording_id=path.stem,
        signal_name=signal_name,
        fault_type=fault,
        fault_size_in=size,
        load=load,
        sensor_location=location,
        sampling_rate_hz=sampling_rate_hz,
    )
    return record, signal


def iter_records(root: Path, sampling_rate_hz: float = 12_000.0) -> Iterator[tuple[CWRURecord, np.ndarray]]:
    """Yield validated CWRU records in stable path order."""

    for path in discover_mat_files(root):
        yield load_record(path, sampling_rate_hz)
