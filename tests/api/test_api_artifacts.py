import json
from pathlib import Path

from fluxguard.api.artifacts import validate_run_artifacts


def test_validate_run_artifacts_accepts_complete_bundle(tmp_path: Path) -> None:
    for name in ("model.pt", "preprocessing.json", "split-manifest.json", "metrics.json"):
        path = tmp_path / name
        if name == "model.pt":
            path.write_bytes(b"checkpoint")
        else:
            path.write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"kind": "anomaly"}), encoding="utf-8")

    valid, reasons = validate_run_artifacts(tmp_path)

    assert valid is True
    assert reasons == ()


def test_validate_run_artifacts_reports_missing_files(tmp_path: Path) -> None:
    valid, reasons = validate_run_artifacts(tmp_path)

    assert valid is False
    assert "missing required artifact: model.pt" in reasons
