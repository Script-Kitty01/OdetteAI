"""Load trained run directories into the FastAPI model registry."""

import json
from pathlib import Path

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
    complete = [
        path.parent
        for path in sorted(models_root.glob("*/*/model.pt"))
        if (path.parent / "config.json").is_file()
        and (path.parent / "preprocessing.json").is_file()
        and (path.parent / "split-manifest.json").is_file()
        and (path.parent / "metrics.json").is_file()
    ]
    return sorted(complete, key=lambda path: path.name)


def load_service_from_runs(models_root: Path) -> FluxGuardService:
    """Load the most recent anomaly, classifier, and RUL runs below *models_root*.

    Only raw-signal 1D classifiers are loaded because the inference API accepts
    raw vibration signals; spectrogram runs remain available for offline use.
    """

    service = FluxGuardService()
    for run_dir in discover_run_dirs(models_root):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        kind = config.get("kind")
        if kind == "anomaly" and service.anomaly_model is None:
            args = config.get("model_args", {})
            model = load_checkpoint(CWRUAutoencoder(**args), run_dir)
            threshold_config = config.get("threshold", {})
            service.anomaly_model = model
            service.anomaly_threshold = AnomalyThreshold(
                threshold_config["mean_error"], threshold_config["std_error"],
                threshold_config.get("multiplier", 3.0),
            ).threshold
        elif kind == "classifier" and service.classifier is None:
            if config.get("representation", "raw") != "raw":
                continue
            model_class = _CLASSIFIERS.get(config.get("model_class", ""))
            if model_class is None:
                continue
            service.classifier = load_checkpoint(model_class(**config.get("model_args", {})), run_dir)
            service.class_names = tuple(config.get("class_names", ()))
        elif kind == "rul" and service.rul_model is None:
            args = dict(config.get("model_args", {}))
            args.setdefault("hidden_size", 64)
            service.rul_model = load_checkpoint(CMapssRULRegressor(**args), run_dir)
    return service
