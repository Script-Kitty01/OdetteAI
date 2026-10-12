# FluxGuard AI Machine Learning Plan

## 1. Purpose

Define the machine-learning implementation plan for FluxGuard AI using the
locally installed benchmark datasets:

- **CWRU Bearing Dataset** for vibration-based anomaly detection and fault
  classification.
- **NASA C-MAPSS Dataset** for multivariate degradation modeling and Remaining
  Useful Life (RUL) estimation.

This document is separate from the repository scaffold plan in the root
`plan.md`. It describes the ML workflow and acceptance criteria. The compact
baseline implementation is now available in `src/fluxguard`.

## 2. ML objectives

### 2.1 CWRU objectives

1. Detect deviations from healthy bearing behavior using an unsupervised model.
2. Classify known bearing faults after an anomaly is detected.
3. Compare raw-signal, engineered-feature, and time-frequency representations.
4. Report results without leakage between related recordings or adjacent
   windows.

### 2.2 NASA C-MAPSS objectives

1. Build engine-cycle sequences from the four standard subsets: FD001, FD002,
   FD003, and FD004.
2. Predict RUL from multivariate sensor histories.
3. Compare performance by subset and operating-condition complexity.
4. Preserve engine-level separation between training and evaluation data.

### 2.3 Explicit non-objective

The first ML release will not claim a validated excess-energy or efficiency
prediction. C-MAPSS does not directly provide a measured power-consumption
target. That capability requires a separate, explicitly defined target and
dataset.

## 3. Data organization and provenance

### 3.1 Input locations

```text
data/raw/
├── cwru/
│   └── *.mat
└── nasa-cmapss/
    ├── C-MAPSS.zip
    └── extracted/
        ├── train_FD001.txt ... train_FD004.txt
        ├── test_FD001.txt ... test_FD004.txt
        ├── RUL_FD001.txt ... RUL_FD004.txt
        └── readme.txt
```

### 3.2 Provenance requirements

Every processed dataset and model artifact must record:

- source dataset and subset;
- source file or engine IDs;
- preprocessing configuration;
- window size, stride, and sampling rate;
- selected channels/sensors;
- random seed;
- train/validation/test split policy;
- package and model version;
- creation timestamp.

Raw benchmark data and generated artifacts remain excluded from Git.

## 4. Stage A — dataset validation and indexing

### CWRU

- Discover all `.mat` files under `data/raw/cwru/`.
- Identify drive-end and fan-end signals using the MATLAB variable names.
- Normalize file metadata into a tabular index.
- Map healthy and fault files to stable labels.
- Preserve motor load, fault location, fault size, sampling rate, and source
  recording where available.
- Fail explicitly when a file has no recognized vibration channel.

### NASA C-MAPSS

- Parse whitespace-separated files with the documented 26-column layout.
- Assign stable names to unit ID, cycle, three operating settings, and 21
  sensors.
- Load RUL targets for the test subsets.
- Derive training RUL as the remaining cycles to the final cycle per engine.
- Validate monotonic cycle order and consistent engine IDs.
- Track FD001–FD004 separately rather than silently merging their distributions.
- Compute per-sensor variance on the training partition before normalization.
  Drop channels below a configured variance threshold and persist the selected
  sensor list. For FD001, the default review list includes Sensors 1, 5, 6,
  10, 16, 18, and 19, but the implementation must verify this from the
  partition instead of assuming the same constant-channel set applies to every
  subset.

### Acceptance criteria

- Every recognized file has a metadata record.
- Invalid files produce actionable validation errors.
- Dataset indexing is deterministic and repeatable.

## 5. Stage B — preprocessing and splitting

### 5.1 CWRU signal conditioning and windowing

- Remove the per-recording DC offset and apply mandatory detrending with
  `scipy.signal.detrend`.
- Apply a configurable high-pass Butterworth filter before window generation
  and feature calculation. The initial default cutoff is **10 Hz**, with filter
  order, cutoff, and sampling rate recorded in preprocessing metadata.
- Validate that the configured cutoff is below the Nyquist frequency.

- Default window length: **2,048 samples**.
- Default overlap: **50%**.
- Make window size, stride, channel, and sampling rate configurable.
- Preserve recording ID and fault label with each window.
- Avoid splitting neighboring windows from one recording across train and test.

### 5.2 NASA sequence construction

- Group rows by engine ID.
- Sort each engine by cycle.
- Build fixed-length sequences with configurable history length and stride.
- Fit scalers only on training engines.
- Apply the same scaler to validation and test engines.
- Support optional RUL clipping, with the cap recorded in metadata.

### 5.3 Split policy

- CWRU: split by source recording/operating condition, not by randomly
  shuffling overlapping windows.
- NASA: split by engine ID; never mix cycles from one engine across splits.
- Use fixed split manifests so repeated runs use the same examples.

### Acceptance criteria

- No source recording or engine ID appears in multiple partitions.
- Scalers and split manifests are persisted.
- A rerun with the same seed produces identical indexes and splits.

## 6. Stage C — signal representations

### 6.1 CWRU time-domain features

For each vibration window, compute:

- mean and standard deviation;
- RMS;
- peak-to-peak;
- absolute peak;
- skewness;
- kurtosis;
- crest factor;
- impulse factor;
- shape/form factor.

Guard against zero RMS and constant windows without returning silent
success-shaped values.

### 6.2 CWRU frequency-domain features

- Apply detrending and configurable windowing before FFT.
- Generate one-sided magnitude or power spectra.
- Record frequency resolution and sampling rate.
- Extract configurable band-energy and dominant-frequency features.
- Add BPFO, BPFI, and BSF features only when bearing geometry parameters are
  explicitly configured.

### 6.3 CWRU time-frequency representations

Implement one baseline representation first:

- **STFT** as the default baseline, with configurable `n_fft`, hop length, and
  output size.

Keep CWT/scalogram generation as a compatible second representation rather
than training multiple image pipelines in the first iteration.

### 6.4 NASA features

- Start with normalized raw sensor sequences.
- Apply the training-only variance thresholding step from Stage A before
  normalization, and persist the resulting channel selection for validation,
  test, inference, and checkpoint loading.
- Preserve operating settings for FD002 and FD004.
- Add rolling statistics only after the raw-sequence baseline is evaluated.

## 7. Stage D — model sequence

### 7.1 Model 1: healthy-only anomaly detector

**Initial architecture:** 1D convolutional autoencoder over normalized CWRU
windows.

Training rules:

- Train only on healthy training windows.
- Use healthy validation windows for threshold calibration.
- Persist model weights, preprocessing metadata, and threshold together.

Anomaly rule:

```text
threshold = mean(healthy_validation_error)
          + 3 * std(healthy_validation_error)
```

The multiplier must remain configurable and must be recorded with the model.

Outputs:

- reconstruction error;
- threshold;
- boolean anomaly decision;
- model/version metadata.

### 7.2 Model 2: CWRU fault classifier

**Initial architecture:** compact 1D CNN baseline. Add SE-TCN only after the
baseline data path and evaluation are stable.

Training rules:

- Train on labeled CWRU windows using recording-aware partitions.
- Report class counts and class imbalance.
- Use class weights or a documented sampler when imbalance requires it.
- Keep fault-label mappings versioned with checkpoints.

Outputs:

- predicted class;
- class probabilities;
- confusion matrix;
- macro and weighted metrics.

### 7.3 Model 3: NASA RUL regressor

**Initial architecture:** LSTM or GRU sequence regressor over normalized sensor
histories.

Training rules:

- Train on engine-level sequences.
- Evaluate separately on FD001–FD004.
- Compare a capped-RUL and uncapped-RUL configuration if both are supported.
- Keep model input sensor order fixed and recorded.

Outputs:

- predicted RUL;
- MAE;
- RMSE;
- score by subset and engine;
- error distribution and worst-engine examples.

## 8. Stage E — training and experiment tracking

Each training run must save:

```text
models/<model-name>/<run-id>/
├── config.json
├── preprocessing.json
├── split-manifest.json
├── metrics.json
├── model.pt
├── model.onnx
└── README.md
```

Required run controls:

- deterministic seed;
- CPU-compatible default configuration;
- configurable batch size, epochs, learning rate, and early stopping;
- checkpointing by validation metric;
- explicit device selection;
- training and validation loss history.

### ONNX export readiness

- Export every completed model to `model.onnx` after checkpoint selection.
- Validate the export with ONNX Runtime or the project-approved ONNX checker
  before marking the run complete.
- Use a dynamic batch axis while preserving the model's fixed feature/channel
  and sequence dimensions.
- Record the input names, output names, opset, dynamic axes, and expected
  tensor shapes in `config.json` or `README.md`.
- Treat export failure as a failed training artifact, not as a successful run
  with a missing deployment file.
- Keep the PyTorch checkpoint as the source-of-truth artifact; ONNX is an
  execution/export artifact for later AMD hardware and quantization workflows.

The first implementation may use JSON artifacts rather than a full experiment
tracking service.

## 9. Stage F — evaluation and quality gates

### CWRU anomaly detection

- healthy false-positive rate;
- anomaly recall on held-out fault recordings;
- precision, recall, F1, and ROC-AUC where scores support it;
- threshold sensitivity analysis.

### CWRU classification

- accuracy;
- macro F1;
- weighted F1;
- per-class precision/recall;
- confusion matrix;
- performance by fault location and load where metadata permits.

### NASA RUL

- MAE;
- RMSE;
- subset-level metrics for FD001–FD004;
- engine-level error distribution;
- early/late-life error analysis.

### Quality gates

1. **Data gate:** schema and provenance checks pass.
2. **Leakage gate:** no source recording or engine crosses partitions.
3. **Baseline gate:** simple baseline metrics are recorded before deep models.
4. **Reproducibility gate:** same configuration and seed reproduce split
   manifests and comparable metrics.
5. **Artifact gate:** checkpoint cannot be loaded without its preprocessing
   metadata.

## 10. Planned implementation order

1. Add dataset index and validation utilities.
2. Add synthetic CWRU `.mat` fixtures and C-MAPSS whitespace fixtures.
3. Add CWRU recording-aware signal conditioning, windowing, and tests.
4. Add NASA parsing, training-only sensor variance pruning, RUL labels, sequence
   generation, and engine-level split tests.
5. Add scaling and persisted preprocessing metadata.
6. Add time-domain and FFT feature extraction. **Completed.**
7. Add the CWRU autoencoder and threshold calibration. **Completed.**
8. Add the CWRU 1D CNN classifier. **Completed.**
9. Add the NASA LSTM/GRU RUL baseline. **Completed.**
10. Add ONNX export, dynamic-batch validation, and artifact loading checks.
    **Export utility completed; full run orchestration remains.**
11. Add evaluation reports and reproducibility checks. **Metrics completed.**
12. Compare baseline configurations before introducing SE-TCN or CWT.
13. Add notebook experiments only as consumers of the reusable package code.

## 11. Deliverables

- Reproducible dataset indexes and split manifests.
- Synthetic parser and split fixtures that prove recording/engine IDs do not
  cross partitions.
- Tested CWRU and NASA preprocessing pipelines.
- Verified CWRU detrending and high-pass conditioning.
- Verified training-only C-MAPSS sensor pruning with persisted channel lists.
- Healthy-only CWRU anomaly model.
- CWRU fault-classification baseline.
- NASA C-MAPSS RUL baseline.
- ONNX exports with validated dynamic batch axes for each completed model.
- Evaluation reports with leakage and provenance evidence.
- Versioned model artifacts with their preprocessing configuration.
- Updated README usage instructions after the ML implementation is complete.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overlapping-window leakage | Split by recording before window generation or group windows by source ID. |
| CWRU file/channel naming variation | Use explicit channel discovery and fail on ambiguity. |
| NASA operating-condition shift | Report FD001–FD004 separately and preserve operating settings. |
| Class imbalance | Report class counts and use documented weighting/sampling. |
| Threshold instability | Calibrate only on healthy validation data and persist the threshold. |
| Unjustified efficiency claims | Defer energy modeling until a measured target is available. |
| Reproducibility drift | Persist seeds, configs, split manifests, and preprocessing metadata. |
