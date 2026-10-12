from pathlib import Path

import numpy as np
import pytest

from fluxguard.datasets.cmapss import (
    COLUMNS,
    CMapssDataError,
    add_training_rul,
    fit_sensor_selection,
    make_engine_sequences,
    read_cmapss,
)


def _row(engine: int, cycle: int, sensor_1: float = 1.0) -> str:
    values = [engine, cycle, 0.1, 0.2, 0.3, sensor_1] + list(range(2, 22))
    return " ".join(str(value) for value in values)


def test_cmapss_parser_assigns_documented_columns(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    path.write_text("\n".join([_row(1, 1), _row(1, 2), _row(2, 1)]))

    frame = read_cmapss(path)

    assert list(frame.columns) == COLUMNS
    assert frame["engine_id"].tolist() == [1, 1, 2]


def test_cmapss_rejects_non_monotonic_cycles(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("\n".join([_row(1, 2), _row(1, 1)]))

    with pytest.raises(CMapssDataError, match="not monotonic"):
        read_cmapss(path)


def test_rul_pruning_and_sequences_are_engine_local(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("\n".join(_row(engine, cycle, float(cycle)) for engine in (1, 2, 3) for cycle in range(1, 5)))
    frame = add_training_rul(read_cmapss(path), cap=2)
    selection = fit_sensor_selection(frame, variance_threshold=0.0)
    sequences, targets, engines = make_engine_sequences(frame, selection.selected, 2)

    assert "sensor_1" in selection.selected
    assert len(sequences) == 9
    assert np.all(targets <= 2)
    assert set(engines) == {1, 2, 3}
