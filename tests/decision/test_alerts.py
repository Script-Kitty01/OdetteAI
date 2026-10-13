from fluxguard.decision.alerts import AlertEngine, AlertSeverity, IncidentStatus


def test_alert_engine_sensor_fault_precedence() -> None:
    engine = AlertEngine()
    incident = engine.evaluate_reading(
        asset_id="bearing_01",
        is_anomaly=False,
        reconstruction_error=0.01,
        threshold=0.05,
        sensor_healthy=False,
        sensor_status="disconnected",
    )
    assert incident is not None
    assert incident.severity == AlertSeverity.WARNING
    assert incident.status == IncidentStatus.ACTIVE
    assert "Sensor" in incident.title


def test_alert_engine_hysteresis_persistence() -> None:
    engine = AlertEngine(persistence_window=3, recovery_window=2)

    # 1st anomaly: below persistence threshold -> no incident yet
    inc1 = engine.evaluate_reading("bearing_02", is_anomaly=True, reconstruction_error=0.08, threshold=0.05)
    assert inc1 is None

    # 2nd anomaly: still below
    inc2 = engine.evaluate_reading("bearing_02", is_anomaly=True, reconstruction_error=0.09, threshold=0.05)
    assert inc2 is None

    # 3rd anomaly: reaches persistence window (3) -> incident created!
    inc3 = engine.evaluate_reading("bearing_02", is_anomaly=True, reconstruction_error=0.10, threshold=0.05)
    assert inc3 is not None
    assert inc3.status == IncidentStatus.ACTIVE

    # Subsequent anomaly updates occurrence count
    inc4 = engine.evaluate_reading("bearing_02", is_anomaly=True, reconstruction_error=0.12, threshold=0.05)
    assert inc4 is not None
    assert inc4.occurrence_count == 2


def test_alert_engine_resolves_sensor_incident_on_recovery() -> None:
    engine = AlertEngine()
    incident = engine.evaluate_reading(
        "bearing_03",
        is_anomaly=False,
        reconstruction_error=0.0,
        threshold=0.1,
        sensor_healthy=False,
        sensor_status="disconnected",
    )
    assert incident is not None

    recovered = engine.evaluate_reading(
        "bearing_03",
        is_anomaly=False,
        reconstruction_error=0.0,
        threshold=0.1,
        sensor_healthy=True,
    )
    assert recovered is None
    assert incident.status == IncidentStatus.RESOLVED
    assert engine.list_incidents(active_only=True) == []
