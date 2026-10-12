"""Minimal inference API for the trained FluxGuard baseline models."""

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fluxguard.api.service import FluxGuardService
from fluxguard.pipelines.train_anomaly import score_autoencoder
from fluxguard.pipelines.train_classifier import predict_classifier
from fluxguard.pipelines.train_rul import predict_rul


class SignalRequest(BaseModel):
    values: list[float] = Field(min_length=8)


class SequenceRequest(BaseModel):
    values: list[list[float]] = Field(min_length=1)


__all__ = ["FluxGuardService", "app", "create_app"]  # FluxGuardService re-exported from fluxguard.api.service


def create_app(
    service: FluxGuardService | None = None, models_root: str | Path | None = None
) -> FastAPI:
    """Build the API app, optionally loading completed runs below *models_root*.

    When neither *service* nor *models_root* is given, the
    ``FLUXGUARD_MODELS_ROOT`` environment variable is consulted; if that is
    also unset the app starts with an empty registry and reports 503s.
    """

    if service is None:
        root = models_root or os.environ.get("FLUXGUARD_MODELS_ROOT")
        if root is not None:
            from fluxguard.api.registry import load_service_from_runs

            service = load_service_from_runs(Path(root))
    registry = service or FluxGuardService()
    app = FastAPI(title="FluxGuard AI", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "models": {
                "anomaly": registry.anomaly_model is not None,
                "classifier": registry.classifier is not None,
                "rul": registry.rul_model is not None,
            },
        }

    @app.post("/api/anomaly/score")
    def anomaly_score(request: SignalRequest) -> dict[str, Any]:
        if registry.anomaly_model is None or registry.anomaly_threshold is None:
            raise HTTPException(status_code=503, detail="anomaly model is not loaded")
        errors = score_autoencoder(registry.anomaly_model, np.asarray([request.values]))
        error = float(errors[0])
        return {"reconstruction_error": error, "threshold": registry.anomaly_threshold, "anomaly": error > registry.anomaly_threshold}

    @app.post("/api/fault/predict")
    def fault_predict(request: SignalRequest) -> dict[str, Any]:
        if registry.classifier is None:
            raise HTTPException(status_code=503, detail="classifier model is not loaded")
        predictions, probabilities = predict_classifier(registry.classifier, np.asarray([request.values]))
        index = int(predictions[0])
        return {
            "class_index": index,
            "class_name": registry.class_names[index] if index < len(registry.class_names) else str(index),
            "probabilities": probabilities[0].tolist(),
        }

    @app.post("/api/rul/predict")
    def rul_predict(request: SequenceRequest) -> dict[str, float]:
        if registry.rul_model is None:
            raise HTTPException(status_code=503, detail="RUL model is not loaded")
        values = np.asarray([request.values], dtype=np.float32)
        return {"predicted_rul": float(predict_rul(registry.rul_model, values)[0])}

    return app


app = create_app()
