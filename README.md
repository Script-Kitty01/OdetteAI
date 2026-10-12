# FluxGuard AI

FluxGuard AI is planned as a machine-learning pipeline for industrial
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
- `src/fluxguard/api/` - FastAPI health and model inference routes
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

## Planned dependencies

The planned implementation will use PyTorch, NumPy, pandas, SciPy, PyWavelets,
scikit-learn, FastAPI, and Uvicorn. Benchmark datasets must be acquired
separately according to their respective terms and placed under `data/raw/`.

## Plans

- [Project scaffold plan](plan.md)
- [Machine-learning plan](plans/machine-learning/plan.md)

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
