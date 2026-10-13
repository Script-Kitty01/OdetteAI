# FluxGuard AI

FluxGuard AI is a local-first machine-learning pipeline for industrial
condition monitoring using the CWRU bearing dataset and NASA C-MAPSS
degradation dataset.

## Current status

The repository contains a runnable local ML pipeline. Dataset ingestion,
preprocessing, signal features, CPU training helpers, evaluation metrics, and
inference API routes are implemented, with end-to-end experiment orchestration
across six configs: CWRU anomaly detection, four CWRU classifier variants
(1D CNN, SE-TCN, STFT, CWT), and C-MAPSS RUL regression. Each run records a
non-deep baseline first (quality gate), persists a full artifact bundle, and
exports ONNX.

## Data Visualization & Statistical Analysis

Visualize raw telemetry across Time-Domain, Frequency-Domain (FFT), STFT Spectrograms, CWT Scalograms, and C-MAPSS multi-sensor degradation trajectories:

```powershell
$env:PYTHONPATH = "src"
python -m fluxguard.evaluation.visualize
```

Outputs are generated in `reports/visualizations/`:
- `cwru_healthy_multidomain.png`: Multi-domain comparison for healthy baseline vibration.
- `cwru_fault_multidomain.png`: Multi-domain comparison for bearing faults.
- `cwru_class_distributions.png`: Fault class distribution, RMS energy, and kurtosis spread.
- `cwru_summary_stats.csv`: Full statistical indicators across all 160 CWRU recordings.
- `cmapss_sensor_trajectories.png`: Degradation trajectories across cycles for multi-sensor channels and RUL curves.

## Industrial Reliability, Sensor Health & Alert Management

FluxGuard AI includes production reliability features:
- **Sensor Health Quality Gate** (`fluxguard.telemetry.quality`): Detects sensor disconnection (zero flatline), stuck-at values, saturation/rail clipping, and invalid samples before models see data, distinguishing sensor failures from machine failures.
- **Incident & Alert Engine** (`fluxguard.decision.alerts`): Uses persistence windows and hysteresis to suppress alert flapping, deduplicate incidents, and assign severity levels (`INFO`, `WARNING`, `CRITICAL`).
- **Open-Set Fault Rejection**: Out-of-distribution detection flags novel and low-confidence predictions as `"UNKNOWN_FAULT"` rather than making high-confidence false assertions.
- **RUL Uncertainty Bounds**: Cycle regression with explicit lower/upper prediction intervals and validity flags.
- **Artifact Validation**: Model runs are checked for required files and valid
  JSON metadata before loading.

Run the production API:
```powershell
$env:PYTHONPATH = "src"
$env:FLUXGUARD_MODELS_ROOT = "models/runs"
python -m uvicorn fluxguard.api.app:app --reload
```
Available endpoints:
- `GET /health` & `GET /ready`: Process liveness and model readiness probes.
- `GET /api/models`: Model registry manifest and capability flags.
- `GET /api/alerts`: Active maintenance incidents and alarm log.
- `POST /api/telemetry`: Unified industrial telemetry ingestion with automated sensor health checks and incident tracking.
- `POST /api/anomaly/score`: Reconstruction error vs. calibrated 3-sigma threshold.
- `POST /api/fault/predict`: Open-set bearing fault classification.
- `POST /api/rul/predict`: Turbofan remaining useful life with uncertainty intervals.

The RUL interval is currently a heuristic estimate, not a calibrated
probabilistic interval. Alerts and prediction history are held in process
memory. MQTT, OPC UA, and Modbus adapters, authentication, and authorization
are outside the current local API scope.

## Run experiments

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m fluxguard.pipelines.run_experiment run --config configs/cwru_anomaly.json
python -m fluxguard.pipelines.run_experiment run --config configs/cwru_classifier_cnn.json
python -m fluxguard.pipelines.run_experiment run --config configs/cwru_classifier_setcn.json
python -m fluxguard.pipelines.run_experiment run --config configs/cwru_classifier_stft.json
python -m fluxguard.pipelines.run_experiment run --config configs/cwru_classifier_cwt.json
python -m fluxguard.pipelines.run_experiment run --config configs/cmapss_rul_fd001.json
python -m fluxguard.pipelines.run_experiment summarize
```

Per-run artifacts land in `models/runs/<experiment>/<timestamp>/` (checkpoint,
`config.json`, preprocessing, split manifest with leakage fingerprint,
`metrics.json` with baselines, ONNX export, figures, `report.md`); the
aggregate table and chart go to `reports/experiment-summary.md/.png`.

## Project layout

- `src/fluxguard/datasets/` - future CWRU and NASA C-MAPSS adapters
- `src/fluxguard/preprocessing/` - future windowing, scaling, and split logic
- `src/fluxguard/features/` - future time, frequency, and time-frequency features
- `src/fluxguard/models/` - autoencoder, CWRU classifier, and RUL baseline models
- `src/fluxguard/pipelines/` - deterministic CPU training and inference helpers
- `src/fluxguard/evaluation/` - anomaly, classification, and regression metrics
- `src/fluxguard/api/` - FastAPI routes, model registry, and artifact validation
- `data/` - local raw, interim, and processed datasets
- `models/` - generated model artifacts
- `configs/` - future experiment configuration files
- `notebooks/` - exploratory notebooks
- `tests/` - tests matching the source package boundaries

## Local setup

```powershell
python -m pip install -e ".[dev]"
```

Run the API locally with trained runs loaded:

```powershell
$env:PYTHONPATH = "src"
$env:FLUXGUARD_MODELS_ROOT = "models/runs"
python -m uvicorn fluxguard.api.app:app --reload
```

`GET /health` reports which models loaded; without `FLUXGUARD_MODELS_ROOT`
(or with an empty runs directory) the inference routes return 503. A
`Dockerfile` is provided for deployment: mount trained runs at
`/models/runs`.

```powershell
docker build -t fluxguard .
docker run -p 8000:8000 -v ${PWD}/models/runs:/models/runs fluxguard
```

## Dependencies and runtime boundaries

The project uses PyTorch, NumPy, pandas, SciPy, PyWavelets, scikit-learn,
FastAPI, Uvicorn, ONNX, and ONNXScript. ONNX Runtime is optional and is needed
only to execute numerical parity checks and ONNX latency benchmarks. Benchmark
datasets must be acquired separately according to their respective terms and
placed under `data/raw/`.

## Plans

- [Project scaffold plan](plan.md)
- [Machine-learning plan](plans/machine-learning/plan.md)
- [Architecture alignment plan](plans/architecture/plan.md)

## Data foundation status

The first ML implementation slice now provides dataset validation and
preprocessing foundations:

- CWRU MAT-file discovery and vibration-channel loading
- CWRU detrending, DC-offset removal, high-pass filtering, and source-aware
  windows
- NASA C-MAPSS parsing, training RUL labels, and engine-local sequences
- Training-only sensor variance selection
- Deterministic group split manifests and training-only scaler metadata
- CWRU time-domain statistical features and one-sided FFT features

Run the focused tests with:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

The current baseline includes STFT/CWT extraction, healthy-only autoencoder
threshold calibration with sensitivity analysis, recording-aware classifier/RUL
training helpers, artifact/ONNX export utilities, SE-TCN and spectrogram model
comparisons, and per-run evaluation reports with leakage evidence.
