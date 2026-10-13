import json
from pathlib import Path

from fastapi.testclient import TestClient

from fluxguard.api.app import create_app
from fluxguard.api.registry import discover_run_dirs, load_service_from_runs
from fluxguard.models.autoencoder import CWRUAutoencoder
from fluxguard.models.classifier import CWRUClassifier
from fluxguard.pipelines.common import append_readme, create_run_dir, save_checkpoint


def _write_anomaly_run(root: Path, run_id: str) -> Path:
    run_dir = create_run_dir(root, "anomaly-demo", run_id)
    save_checkpoint(
        CWRUAutoencoder(32), run_dir,
        {
            "kind": "anomaly",
            "model_class": "CWRUAutoencoder",
            "model_args": {"input_length": 32},
            "threshold": {"mean_error": 0.1, "std_error": 0.01, "multiplier": 3.0},
        },
        {"window_size": 32}, {"train": ["a"], "validation": ["b"], "test": ["c"]}, {"loss": 0.1},
    )
    append_readme(run_dir, "anomaly", "test run")
    return run_dir


def _write_classifier_run(root: Path, run_id: str) -> Path:
    run_dir = create_run_dir(root, "classifier-demo", run_id)
    save_checkpoint(
        CWRUClassifier(3, 32), run_dir,
        {
            "kind": "classifier",
            "model_class": "CWRUClassifier",
            "model_args": {"class_count": 3, "input_length": 32},
            "class_names": ["healthy", "outer_race_0.007", "inner_race_0.014"],
            "representation": "raw",
        },
        {"window_size": 32}, {"train": ["a"], "validation": ["b"], "test": ["c"]}, {"accuracy": 1.0},
    )
    append_readme(run_dir, "classifier", "test run")
    return run_dir


def test_discover_run_dirs_finds_complete_runs(tmp_path: Path) -> None:
    _write_anomaly_run(tmp_path, "run-1")
    incomplete = create_run_dir(tmp_path, "broken-demo", "run-2")
    (incomplete / "model.pt").write_bytes(b"")
    found = discover_run_dirs(tmp_path)
    assert [path.name for path in found] == ["run-1"]
    assert all((path / "config.json").is_file() for path in found)


def test_load_service_from_runs_builds_working_endpoints(tmp_path: Path) -> None:
    _write_anomaly_run(tmp_path, "run-1")
    _write_classifier_run(tmp_path, "run-1")
    service = load_service_from_runs(tmp_path)
    assert service.anomaly_model is not None
    assert service.anomaly_threshold is not None
    assert service.classifier is not None
    assert service.class_names == ("healthy", "outer_race_0.007", "inner_race_0.014")
    assert service.rul_model is None

    client = TestClient(create_app(service))
    health = client.get("/health").json()
    assert health["models"] == {"anomaly": True, "classifier": True, "rul": False}

    anomaly = client.post("/api/anomaly/score", json={"values": [0.0] * 32})
    assert anomaly.status_code == 200
    assert "reconstruction_error" in anomaly.json()

    fault = client.post("/api/fault/predict", json={"values": [0.0] * 32})
    assert fault.status_code == 200
    assert fault.json()["class_name"] in ("healthy", "outer_race_0.007", "inner_race_0.014", "UNKNOWN_FAULT")


def test_load_service_selects_newest_run_and_retains_manifest(tmp_path: Path) -> None:
    older = _write_anomaly_run(tmp_path, "20260101T000000Z")
    newer = _write_anomaly_run(tmp_path, "20260102T000000Z")
    older_config = json.loads((older / "config.json").read_text(encoding="utf-8"))
    newer_config = json.loads((newer / "config.json").read_text(encoding="utf-8"))
    older_config["threshold"]["mean_error"] = 0.9
    newer_config["threshold"]["mean_error"] = 0.7
    older.joinpath("config.json").write_text(json.dumps(older_config), encoding="utf-8")
    newer.joinpath("config.json").write_text(json.dumps(newer_config), encoding="utf-8")

    service = load_service_from_runs(tmp_path)
    assert service.anomaly_threshold == 0.73
    runs = service.manifest["anomaly-demo"]
    assert [run["run_id"] for run in runs] == ["20260102T000000Z", "20260101T000000Z"]
    assert runs[0]["selected"] is True
    assert runs[1]["selected"] is False


def test_create_app_with_models_root_env(tmp_path: Path, monkeypatch) -> None:
    run_dir = _write_anomaly_run(tmp_path, "run-1")
    config_path = run_dir / "config.json"
    assert json.loads(config_path.read_text(encoding="utf-8"))["kind"] == "anomaly"
    monkeypatch.setenv("FLUXGUARD_MODELS_ROOT", str(tmp_path))
    client = TestClient(create_app())
    assert client.get("/health").json()["models"]["anomaly"] is True


def test_missing_models_root_keeps_empty_registry(tmp_path: Path) -> None:
    service = load_service_from_runs(tmp_path / "does-not-exist")
    assert service.anomaly_model is None
    assert service.classifier is None
    assert service.rul_model is None
