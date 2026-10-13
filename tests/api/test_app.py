from fastapi.testclient import TestClient

from fluxguard.api.app import create_app


def test_health_endpoint_reports_unloaded_models() -> None:
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_endpoints_report_explicit_unavailable_status() -> None:
    client = TestClient(create_app())
    response = client.post("/api/anomaly/score", json={"values": [0.0] * 32})
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]


def test_ready_and_models_endpoints() -> None:
    client = TestClient(create_app())
    ready_res = client.get("/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["ready"] is False

    models_res = client.get("/api/models")
    assert models_res.status_code == 200
    assert "capabilities" in models_res.json()


def test_telemetry_ingest_sensor_disconnected() -> None:
    client = TestClient(create_app())
    # Disconnected flatline sensor
    res = client.post(
        "/api/telemetry",
        json={"asset_id": "test_pump_1", "values": [0.0] * 64},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sensor_health"]["is_valid"] is False
    assert data["sensor_health"]["status"] == "disconnected"
    assert data["incident"] is not None
    assert "Sensor" in data["incident"]["title"]
