"""Incident and alert management engine with hysteresis and deduplication."""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class MaintenanceIncident:
    """An actionable maintenance incident record."""

    incident_id: str
    asset_id: str
    severity: AlertSeverity
    status: IncidentStatus
    title: str
    details: str
    created_at: str
    updated_at: str
    evidence: dict[str, Any] = field(default_factory=dict)
    occurrence_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "asset_id": self.asset_id,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "details": self.details,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "occurrence_count": self.occurrence_count,
            "evidence": self.evidence,
        }


class AlertEngine:
    """Hysteresis-backed alert manager to avoid flapping and deduplicate events."""

    def __init__(
        self,
        persistence_window: int = 3,
        recovery_window: int = 2,
    ) -> None:
        self.persistence_window = persistence_window
        self.recovery_window = recovery_window
        self._history: dict[str, deque[bool]] = {}
        self._active_incidents: dict[str, MaintenanceIncident] = {}
        self._incident_log: list[MaintenanceIncident] = []

    def evaluate_reading(
        self,
        asset_id: str,
        is_anomaly: bool,
        reconstruction_error: float,
        threshold: float,
        fault_type: str | None = None,
        fault_confidence: float = 0.0,
        rul_estimate: float | None = None,
        sensor_healthy: bool = True,
        sensor_status: str = "healthy",
    ) -> MaintenanceIncident | None:
        """Process a condition-monitoring result through hysteresis state machine."""
        now = datetime.now(UTC).isoformat()

        # Handle sensor-level failure first (distinguishing sensor fault from machine fault)
        if not sensor_healthy:
            incident_key = f"{asset_id}:sensor_fault"
            if incident_key in self._active_incidents:
                inc = self._active_incidents[incident_key]
                inc.occurrence_count += 1
                inc.updated_at = now
                return inc

            incident = MaintenanceIncident(
                incident_id=f"INC-{uuid.uuid4().hex[:8].upper()}",
                asset_id=asset_id,
                severity=AlertSeverity.WARNING,
                status=IncidentStatus.ACTIVE,
                title=f"Sensor Health Degradation on {asset_id}",
                details=f"Sensor reported status '{sensor_status}'. Verify physical wiring or connection before servicing machine.",
                created_at=now,
                updated_at=now,
                evidence={"sensor_status": sensor_status},
            )
            self._active_incidents[incident_key] = incident
            self._incident_log.append(incident)
            return incident

        sensor_incident_key = f"{asset_id}:sensor_fault"
        if sensor_incident_key in self._active_incidents:
            sensor_incident = self._active_incidents.pop(sensor_incident_key)
            sensor_incident.status = IncidentStatus.RESOLVED
            sensor_incident.updated_at = now

        # Rolling hysteresis for mechanical/thermodynamic anomaly
        if asset_id not in self._history:
            self._history[asset_id] = deque(maxlen=self.persistence_window)

        hist = self._history[asset_id]
        hist.append(is_anomaly)

        # Trigger alert only if anomaly persists across required window
        consecutive_anomalies = sum(1 for x in hist if x)
        incident_key = f"{asset_id}:machine_fault"

        if consecutive_anomalies >= self.persistence_window:
            severity = AlertSeverity.CRITICAL if (rul_estimate is not None and rul_estimate < 30) or (fault_confidence > 0.8) else AlertSeverity.WARNING
            
            title = f"Persistent Anomaly on {asset_id}"
            if fault_type and fault_type.upper() != "HEALTHY":
                title += f" (Suspected: {fault_type.upper()})"

            details = f"Anomaly score {reconstruction_error:.4f} exceeded threshold {threshold:.4f} across {consecutive_anomalies} consecutive windows."
            if rul_estimate is not None:
                details += f" Estimated RUL: {rul_estimate:.1f} cycles."

            if incident_key in self._active_incidents:
                inc = self._active_incidents[incident_key]
                inc.occurrence_count += 1
                inc.updated_at = now
                inc.severity = severity
                inc.details = details
                inc.evidence.update(
                    {
                        "reconstruction_error": reconstruction_error,
                        "threshold": threshold,
                        "fault_type": fault_type,
                        "fault_confidence": fault_confidence,
                        "rul_estimate": rul_estimate,
                    }
                )
                return inc

            incident = MaintenanceIncident(
                incident_id=f"INC-{uuid.uuid4().hex[:8].upper()}",
                asset_id=asset_id,
                severity=severity,
                status=IncidentStatus.ACTIVE,
                title=title,
                details=details,
                created_at=now,
                updated_at=now,
                evidence={
                    "reconstruction_error": reconstruction_error,
                    "threshold": threshold,
                    "fault_type": fault_type,
                    "fault_confidence": fault_confidence,
                    "rul_estimate": rul_estimate,
                },
            )
            self._active_incidents[incident_key] = incident
            self._incident_log.append(incident)
            return incident

        # Auto-resolve or clear if consecutive normal readings observed
        consecutive_normals = sum(1 for x in hist if not x)
        if consecutive_normals >= self.recovery_window and incident_key in self._active_incidents:
            inc = self._active_incidents.pop(incident_key)
            inc.status = IncidentStatus.RESOLVED
            inc.updated_at = now

        return None

    def list_incidents(self, active_only: bool = False) -> list[MaintenanceIncident]:
        if active_only:
            return list(self._active_incidents.values())
        return list(self._incident_log)
