"""Minimal inference API for the trained FluxGuard baseline models."""

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fluxguard.models.autoencoder import CWRUAutoencoder
from fluxguard.models.classifier import CWRUClassifier
from fluxguard.models.rul import CMapssRULRegressor
from fluxguard.pipelines.train_anomaly import score_autoencoder
from fluxguard.pipelines.train_classifier import predict_classifier
from fluxguard.pipelines.train_rul import predict_rul


class SignalRequest(BaseModel):
    values: list[float] = Field(min_length=8)


class SequenceRequest(BaseModel):
    values: list[list[float]] = Field(min_length=1)


class FluxGuardService:
    """Model registry used by the API and straightforward to replace in deployment."""

    def __init__(
        self,
        anomaly_model: CWRUAutoencoder | None = None,
        anomaly_threshold: float | None = None,
        classifier: CWRUClassifier | None = None,
        class_names: tuple[str, ...] = (),
        rul_model: CMapssRULRegressor | None = None,
    ) -> None:
        self.anomaly_model = anomaly_model
        self.anomaly_threshold = anomaly_threshold
        self.classifier = classifier
        self.class_names = class_names
        self.rul_model = rul_model


def create_app(service: FluxGuardService | None = None) -> FastAPI:
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
