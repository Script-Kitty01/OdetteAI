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
