from pathlib import Path

import numpy as np
from scipy.io import savemat

from fluxguard.datasets.cwru import load_record


def test_cwru_loads_drive_end_signal_and_healthy_metadata(tmp_path: Path) -> None:
    path = tmp_path / "97.mat"
    savemat(path, {"X097_DE_time": np.arange(32, dtype=float)})

    record, values = load_record(path)

    assert record.fault_type == "healthy"
    assert record.sensor_location == "drive_end"
    assert values.shape == (32,)
