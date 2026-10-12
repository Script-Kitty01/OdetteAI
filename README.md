# FluxGuard AI

FluxGuard AI is planned as a machine-learning pipeline for industrial
condition monitoring using the CWRU bearing dataset and NASA C-MAPSS
degradation dataset.

## Current status

The repository contains a runnable local ML baseline. Dataset ingestion,
preprocessing, signal features, CPU training helpers, evaluation metrics, and
inference API routes are implemented; production deployment and advanced
architectures remain future work.

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

Run the API locally with:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn fluxguard.api.app:app --reload
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

The current baseline includes STFT extraction, healthy-only autoencoder
threshold calibration, recording-aware classifier/RUL training helpers, and
artifact/ONNX export utilities. Full dataset-scale experiment orchestration,
model artifact loading, and advanced SE-TCN/CWT comparisons remain follow-up
work.
