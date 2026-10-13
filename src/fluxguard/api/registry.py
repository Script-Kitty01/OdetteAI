"""Load trained run directories into the FastAPI model registry."""

import json
from pathlib import Path

from fluxguard.api.artifacts import validate_run_artifacts
from fluxguard.api.service import FluxGuardService
from fluxguard.models.autoencoder import AnomalyThreshold, CWRUAutoencoder
from fluxguard.models.classifier import CWRUClassifier
from fluxguard.models.rul import CMapssRULRegressor
from fluxguard.models.spectrogram import SpectrogramCNN
from fluxguard.models.tcn import SETCNClassifier
from fluxguard.pipelines.common import load_checkpoint

_CLASSIFIERS = {
    "CWRUClassifier": CWRUClassifier,
    "SETCNClassifier": SETCNClassifier,
    "SpectrogramCNN": SpectrogramCNN,
}


def discover_run_dirs(models_root: Path) -> list[Path]:
    """Return run directories that contain a complete artifact bundle."""

    if not models_root.exists():
        return []
    complete = []
    for model_path in models_root.glob("*/*/model.pt"):
        run_dir = model_path.parent
        valid, _ = validate_run_artifacts(run_dir)
        if valid:
            complete.append(run_dir)
    return sorted(complete, key=lambda path: path.name, reverse=True)


def load_service_from_runs(models_root: Path) -> FluxGuardService:
    """Load the most recent anomaly, classifier, and RUL runs below *models_root*.

    Only raw-signal 1D classifiers are loaded because the inference API accepts
    raw vibration signals; spectrogram runs remain available for offline use.
    """

    service = FluxGuardService()
    discovered = discover_run_dirs(models_root)
    selected: set[str] = set()
    selected_runs: dict[str, str] = {}
    for run_dir in discovered:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        kind = config.get("kind")
        service.manifest.setdefault(run_dir.parent.name, []).append(
            {
                "run_id": run_dir.name,
                "kind": kind,
                "has_onnx": (run_dir / "model.onnx").is_file(),
                "config": config,
                "selected": False,
            }
        )
        if kind in selected:
            continue
        if kind == "anomaly" and service.anomaly_model is None:
            args = config.get("model_args", {})
            model = load_checkpoint(CWRUAutoencoder(**args), run_dir)
            threshold_config = config.get("threshold", {})
            service.anomaly_model = model
            service.anomaly_threshold = AnomalyThreshold(
                threshold_config["mean_error"], threshold_config["std_error"],
                threshold_config.get("multiplier", 3.0),
            ).threshold
            selected.add(kind)
            selected_runs[kind] = run_dir.name
        elif kind == "classifier" and service.classifier is None:
            if config.get("representation", "raw") != "raw":
                continue
            model_class = _CLASSIFIERS.get(config.get("model_class", ""))
            if model_class is None:
                continue
            service.classifier = load_checkpoint(model_class(**config.get("model_args", {})), run_dir)
            service.class_names = tuple(config.get("class_names", ()))
            selected.add(kind)
            selected_runs[kind] = run_dir.name
        elif kind == "rul" and service.rul_model is None:
            args = dict(config.get("model_args", {}))
            args.setdefault("hidden_size", 64)
            service.rul_model = load_checkpoint(CMapssRULRegressor(**args), run_dir)
            selected.add(kind)
            selected_runs[kind] = run_dir.name
    for runs in service.manifest.values():
        for run in runs:
            run["selected"] = selected_runs.get(run["kind"]) == run["run_id"]
    return service
