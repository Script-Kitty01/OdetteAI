# FluxGuard Architecture Alignment Plan

## Goal

Align the architecture documentation with the implemented FluxGuard runtime
and close the reliability gaps identified during review.

## Implementation scope

1. **Model registry correctness**
   - Select the newest complete run for each loadable model kind.
   - Preserve all discovered runs in the registry manifest.
   - Add explicit artifact validation before a run is considered loadable.

2. **Benchmark truthfulness**
   - Report ONNX parity as `skipped` when `onnxruntime` is unavailable.
   - Never report a skipped comparison as a successful parity result.

3. **Alert lifecycle**
   - Resolve active sensor incidents after the configured recovery window of
     healthy readings.
   - Preserve the resolved incident in the incident history.

4. **Architecture documentation**
   - Add reliability and decision components.
   - Add telemetry quality-gate and alert flow.
   - Add artifact validation and model-registry boundaries.
   - Add offline visualization and reporting.
   - Document current operational boundaries and optional ONNX Runtime status.
   - Include all current API endpoints in the component inventory.

5. **Verification**
   - Add focused tests for newest-run selection, artifact validation,
     benchmark skip semantics, and sensor-incident recovery.
   - Run the full test suite, Ruff, and whitespace checks.

## Acceptance criteria

- The registry loads the newest complete compatible run, not the oldest.
- Invalid or incomplete artifact bundles are excluded with an actionable
  validation reason.
- ONNX parity results distinguish `passed`, `failed`, and `skipped`.
- Sensor incidents transition to `RESOLVED` after healthy recovery readings.
- The architecture diagram renders with the required Mermaid-safe syntax.
- README and architecture documentation describe the current implementation
  rather than planned functionality.
- All tests and lint checks pass before committing and pushing.

## Explicit boundaries

- No database, authentication, broker, or external industrial protocol is
  added in this change.
- Incident persistence remains in memory.
- RUL intervals remain heuristic and are documented as such.
- ONNX Runtime remains optional unless the dependency is explicitly installed.
