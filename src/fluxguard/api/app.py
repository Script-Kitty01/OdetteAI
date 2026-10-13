"""Industrial inference and telemetry API for FluxGuard AI."""

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
from fluxguard.telemetry.quality import check_signal_quality


class SignalRequest(BaseModel):
    values: list[float] = Field(min_length=8)
    asset_id: str | None = Field(default="bearing_default")


class SequenceRequest(BaseModel):
    values: list[list[float]] = Field(min_length=1)
    asset_id: str | None = Field(default="engine_default")


class TelemetryIngestRequest(BaseModel):
    asset_id: str
    values: list[float] = Field(min_length=8)
    saturation_limit: float | None = None


__all__ = ["FluxGuardService", "app", "create_app"]


def create_app(
    service: FluxGuardService | None = None, models_root: str | Path | None = None
) -> FastAPI:
    """Build the API app, optionally loading completed runs below *models_root*."""
    if service is None:
        root = models_root or os.environ.get("FLUXGUARD_MODELS_ROOT")
        if root is not None:
            from fluxguard.api.registry import load_service_from_runs

            service = load_service_from_runs(Path(root))
    registry = service or FluxGuardService()
    app = FastAPI(title="FluxGuard AI", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Basic process liveness check."""
        return {
            "status": "ok",
            "models": {
                "anomaly": registry.anomaly_model is not None,
                "classifier": registry.classifier is not None,
                "rul": registry.rul_model is not None,
            },
        }

    @app.get("/ready")
    def readiness() -> dict[str, Any]:
        """Readiness probe checking if required models are loaded into memory."""
        is_ready = bool(registry.anomaly_model and registry.classifier)
        return {
            "ready": is_ready,
            "loaded_models": {
                "anomaly_model": registry.anomaly_model is not None,
                "classifier": registry.classifier is not None,
                "rul_model": registry.rul_model is not None,
            },
            "active_incidents_count": len(registry.alert_engine.list_incidents(active_only=True)),
        }

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        """Return loaded models, artifact versions, and deployment capabilities."""
        return {
            "models": registry.manifest,
            "capabilities": {
                "anomaly_detection": registry.anomaly_model is not None,
                "fault_classification": registry.classifier is not None,
                "rul_prediction": registry.rul_model is not None,
                "open_set_rejection": True,
                "sensor_health_monitoring": True,
            },
        }

    @app.get("/api/alerts")
    def list_alerts(active_only: bool = True) -> dict[str, Any]:
        """Retrieve maintenance incidents and alerts from the incident engine."""
        incidents = registry.alert_engine.list_incidents(active_only=active_only)
        return {"incidents": [inc.as_dict() for inc in incidents]}

    @app.post("/api/anomaly/score")
    def anomaly_score(request: SignalRequest) -> dict[str, Any]:
        if registry.anomaly_model is None or registry.anomaly_threshold is None:
            raise HTTPException(status_code=503, detail="anomaly model is not loaded")
        errors = score_autoencoder(registry.anomaly_model, np.asarray([request.values]))
        error = float(errors[0])
        is_anomaly = error > registry.anomaly_threshold
        return {
            "reconstruction_error": error,
            "threshold": registry.anomaly_threshold,
            "anomaly": is_anomaly,
        }

    @app.post("/api/fault/predict")
    def fault_predict(request: SignalRequest) -> dict[str, Any]:
        if registry.classifier is None:
            raise HTTPException(status_code=503, detail="classifier model is not loaded")
        predictions, probabilities = predict_classifier(registry.classifier, np.asarray([request.values]))
        index = int(predictions[0])
        probs = probabilities[0]
        max_prob = float(probs[index])

        # Open-set rejection: if confidence is below calibrated threshold, report UNKNOWN_FAULT
        is_ood = max_prob < registry.min_confidence
        predicted_name = (
            "UNKNOWN_FAULT"
            if is_ood
            else (registry.class_names[index] if index < len(registry.class_names) else str(index))
        )

        return {
            "class_index": -1 if is_ood else index,
            "class_name": predicted_name,
            "confidence": max_prob,
            "is_out_of_distribution": is_ood,
            "probabilities": probs.tolist(),
        }

    @app.post("/api/rul/predict")
    def rul_predict(request: SequenceRequest) -> dict[str, Any]:
        if registry.rul_model is None:
            raise HTTPException(status_code=503, detail="RUL model is not loaded")
        values = np.asarray([request.values], dtype=np.float32)
        base_rul = float(predict_rul(registry.rul_model, values)[0])

        # Uncertainty intervals (standard heuristic bound of +-10% or +-5 cycles)
        margin = max(5.0, base_rul * 0.12)
        return {
            "predicted_rul": base_rul,
            "unit": "cycles",
            "uncertainty_interval": {
                "lower_bound": max(0.0, base_rul - margin),
                "upper_bound": base_rul + margin,
            },
            "is_reliable": base_rul > 5.0,
        }

    @app.post("/api/telemetry")
    def ingest_telemetry(request: TelemetryIngestRequest) -> dict[str, Any]:
        """Unified ingestion pipeline: checks sensor health, scores anomaly, evaluates faults and alerts."""
        values = np.asarray(request.values, dtype=np.float64)

        # 1. Sensor Health Quality Gate
        health_report = check_signal_quality(values, saturation_limit=request.saturation_limit)

        error_val = 0.0
        is_anomaly = False
        threshold_val = registry.anomaly_threshold or 0.0
        fault_name = None
        fault_conf = 0.0

        if health_report.is_valid and registry.anomaly_model and registry.anomaly_threshold:
            errors = score_autoencoder(registry.anomaly_model, np.asarray([request.values]))
            error_val = float(errors[0])
            is_anomaly = error_val > registry.anomaly_threshold

            if registry.classifier:
                predictions, probabilities = predict_classifier(registry.classifier, np.asarray([request.values]))
                index = int(predictions[0])
                probs = probabilities[0]
                fault_conf = float(probs[index])
                fault_name = (
                    "UNKNOWN_FAULT"
                    if fault_conf < registry.min_confidence
                    else (registry.class_names[index] if index < len(registry.class_names) else str(index))
                )

        # 2. Evaluate alert through Incident Engine
        incident = registry.alert_engine.evaluate_reading(
            asset_id=request.asset_id,
            is_anomaly=is_anomaly,
            reconstruction_error=error_val,
            threshold=threshold_val,
            fault_type=fault_name,
            fault_confidence=fault_conf,
            sensor_healthy=health_report.is_valid,
            sensor_status=health_report.status.value,
        )

        return {
            "asset_id": request.asset_id,
            "sensor_health": health_report.as_dict(),
            "analysis": {
                "reconstruction_error": error_val,
                "threshold": threshold_val,
                "is_anomaly": is_anomaly,
                "fault_prediction": fault_name,
                "fault_confidence": fault_conf,
            },
            "incident": incident.as_dict() if incident else None,
        }

    return app


app = create_app()
