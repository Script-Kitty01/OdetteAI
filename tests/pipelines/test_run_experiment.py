import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from fluxguard.pipelines.run_experiment import load_experiment_config, main, run_cwru_anomaly


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_experiment_config_reads_valid_file(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "name": "demo", "kind": "cwru-anomaly", "raw_dir": "data/raw/cwru",
            "seed": 7, "training": {"epochs": 2},
        },
    )
    config = load_experiment_config(path)
    assert config.name == "demo"
    assert config.seed == 7
    assert config.training == {"epochs": 2}
    assert config.signal_config.window_size == 2048


def test_load_experiment_config_rejects_unknown_kind(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"name": "demo", "kind": "nope", "raw_dir": "x"})
    with pytest.raises(ValueError, match="kind"):
        load_experiment_config(path)


def test_load_experiment_config_rejects_missing_keys(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"name": "demo"})
    with pytest.raises(ValueError, match="missing keys"):
        load_experiment_config(path)


def _write_synthetic_cwru(root: Path) -> None:
    rng = np.random.default_rng(0)
    # Stems 97-100 are recognized as healthy; other numeric stems without
    # fault markers are treated as faults. 4 healthy + 8 fault recordings
    # give the group split enough room to populate every partition.
    for stem in ("97", "98", "99", "100"):
        signal = rng.normal(0, 0.1, size=(4096,))
        savemat(root / f"{stem}.mat", {"X097_DE_time": signal})
    for stem in ("105", "106", "107", "108", "109", "110", "111", "112"):
        signal = rng.normal(0, 0.1, size=(4096,))
        signal += 2.0 * np.sin(2 * np.pi * 300 * np.arange(4096) / 1000)
        savemat(root / f"{stem}.mat", {"X097_DE_time": signal})


def test_run_cwru_anomaly_on_synthetic_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "cwru"
    raw_dir.mkdir()
    _write_synthetic_cwru(raw_dir)
    config = load_experiment_config(
        _write_config(
            tmp_path,
            {
                "name": "synthetic-anomaly", "kind": "cwru-anomaly", "raw_dir": str(raw_dir),
                "output_root": str(tmp_path / "runs"),
                "seed": 42,
                "signal": {"window_size": 256, "overlap": 0.5},
                "model": {"max_windows_per_recording": 4},
                "training": {"epochs": 1, "batch_size": 8},
            },
        )
    )
    run_dir = run_cwru_anomaly(config)
    assert (run_dir / "model.pt").is_file()
    assert (run_dir / "model.onnx").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "score-distributions.png").is_file()
    assert (run_dir / "threshold-sensitivity.png").is_file()


def test_main_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
