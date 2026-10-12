# High-Level Design (HLD) — FluxGuard AI

## 1. Executive Summary & System Objectives

**FluxGuard AI** is an industrial-grade machine learning platform engineered for Condition-Based Maintenance (CBM), predictive anomaly detection, and equipment Remaining Useful Life (RUL) estimation. The system operates on edge gateways and industrial computing hardware to monitor high-stress rotating machinery and aerospace thermodynamic assets.

### 1.1 Dual Failure Domain Architecture
Industrial equipment degrades across distinct mechanical and thermodynamic regimes. FluxGuard AI targets two primary failure domains using established reference benchmarks:

1. **Mechanical High-Frequency Vibration (CWRU Bearing Dataset)**:
   - High-frequency 1D accelerometer telemetry (12 kHz and 48 kHz sampling rates).
   - Monitors drive-end, fan-end, and base accelerometer channels.
   - Diagnoses bearing defects across outer race, inner race, and ball elements at multiple defect diameters (0.007", 0.014", 0.021").
2. **Multivariate Thermodynamic Degradation (NASA C-MAPSS Turbofan Dataset)**:
   - Multi-sensor temporal time-series (21 sensor channels across temperatures, pressures, and rotational speeds, plus 3 operational settings).
   - Monitors engine fleet degradation across sub-datasets FD001 through FD004.
   - Predicts asset Remaining Useful Life (RUL) in cycles.

### 1.2 Core Architectural Objectives
- **Zero-Shot Novelty Detection**: Detect unmodeled mechanical faults via an unsupervised autoencoder trained strictly on healthy baseline signals.
- **Strict Anti-Leakage Partitioning**: Enforce recording-level disjoint splits for CWRU and engine-level disjoint splits for C-MAPSS to prevent artificial accuracy inflation.
- **Multi-Representation Diagnostics**: Support raw 1D time-series, statistical time-domain features, discrete Fourier spectra, Short-Time Fourier Transforms (STFT), and Continuous Wavelet Transforms (CWT).
- **Edge Deployment & Hardware Readiness**: Export all verified models to ONNX with validated dynamic batch axes, ready for AMD Quark quantization and edge execution.
- **Self-Contained Auditability**: Persist cryptographic split manifests, preprocessing parameters, baseline comparisons, and diagnostic plots with every training run.

### 1.3 Explicit Non-Objectives
- **Unverified Energy Waste Estimation**: Predicting exact excess energy consumption ($\Delta E_{\text{waste}}$) is explicitly deferred until real measured electrical load data is available. C-MAPSS does not provide ground-truth power measurements.

---

## 2. System Context & Operational Flow

FluxGuard AI operates in both an **offline experimentation/training loop** and an **online real-time inference loop**.

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    subgraph DataSources["Industrial Telemetry Sources"]
        BearingSensors["Vibration Accelerometers - 12kHz / 48kHz"]
        EngineSensors["Turbofan Multi-Sensors - 21 Channels"]
        ArchiveStore[("Raw Archive Store - CWRU / C-MAPSS")]
    end

    subgraph CorePlatform["FluxGuard AI Platform"]
        IngestionMod["Data Ingestion and Integrity Validation"]
        PreprocMod["Signal Conditioning and Windowing Engine"]
        SplitMod["Anti-Leakage Split Engine - SHA-256 Manifest"]
        ModelZooMod["Deep Learning Model Zoo"]
        EvalMod["Evaluation Gates and Baseline Check"]
        ExportMod["ONNX Exporter and Artifact Packager"]
    end

    subgraph ServingEnv["Deployment and Serving"]
        Artifacts[("Model Registry - models/runs")]
        FastApiServer["FastAPI Inference Service - Uvicorn"]
        IndustrialClient["Industrial Gateway / SCADA Client"]
        Dashboard["Experiment and Health Dashboard"]
    end

    BearingSensors -->|"OPC UA / Modbus"| IngestionMod
    EngineSensors -->|"Telemetry Stream"| IngestionMod
    ArchiveStore -->|"Batch Files"| IngestionMod
    IngestionMod -->|"Validated Arrays"| PreprocMod
    PreprocMod -->|"Conditioned Windows"| SplitMod
    SplitMod -->|"Disjoint Partitions"| ModelZooMod
    ModelZooMod -->|"Validation Loss"| EvalMod
    EvalMod -->|"Quality Gate Pass"| ExportMod
    ExportMod -->|"Artifact Package"| Artifacts
    Artifacts -->|"Load Active Models"| FastApiServer
    FastApiServer -->|"REST JSON API"| IndustrialClient
    Artifacts -->|"Reports and Metrics"| Dashboard
```

---

## 3. Data Ingestion & Preprocessing Subsystem

The data subsystem ensures that raw physical signals are normalized, validated, and partitioned without cross-sample data leakage.

### 3.1 CWRU Vibration Signal Pipeline
1. **Discovery & Validation**:
   - Discovers MATLAB `.mat` files in `data/raw/cwru/`.
   - Identifies vibration channels: `X*_DE_time` (Drive-End), `X*_FE_time` (Fan-End), and `X*_BA_time` (Base).
   - Extracts motor load (0-3 HP), defect location, and fault size.
   - Corrupt files (such as truncated MAT files) trigger an explicit `CWRUDataError` instead of silent failures.
2. **Signal Conditioning**:
   - **Linear Detrending**: Removes sensor drift and DC offsets using `scipy.signal.detrend`.
   - **Zero-Phase High-Pass Butterworth Filter**: Removes low-frequency structural noise using a 4th-order filter with a default 10 Hz cutoff frequency (verified below the Nyquist limit).
3. **Sliding Window Generation**:
   - Slices continuous 1D signals into fixed windows of 2,048 samples with 50% overlap (stride = 1,024 samples).
   - Preserves source recording ID, physical defect label, and sample boundaries for traceability.

### 3.2 NASA C-MAPSS Turbofan Pipeline
1. **Schema Validation**:
   - Parses whitespace-delimited logs containing 26 columns: `engine_id`, `cycle`, 3 operational settings, and 21 sensor channels.
   - Enforces strictly monotonic cycle ordering per engine.
2. **RUL Label Construction**:
   - Derives training RUL: $\text{RUL}_{i,t} = \max(\text{cycle}_i) - \text{cycle}_{i,t}$.
   - Applies piecewise linear degradation clipping (default cap: 125 cycles) to model stable initial health followed by progressive wear.
3. **Training-Only Sensor Variance Pruning**:
   - Computes variance across sensors on training partitions only.
   - Drops non-informative invariant channels (variance $\le 10^{-12}$). On FD001, sensors 1, 5, 6, 10, 16, 18, and 19 are verified invariant, reducing input width from 21 to 14-15 active sensors.
4. **Sequence Construction**:
   - Groups data strictly by `engine_id` to form multivariate time-series tensors of shape `(batch, 30, num_sensors)`.

### 3.3 Leakage Prevention & Deterministic Splitting
Data leakage between training and testing partitions is the leading cause of artificially inflated performance in predictive maintenance literature.
- **CWRU Group Splitting**: Partitioning occurs at the `source_recording` level. Overlapping windows from the same recording are never split across train, validation, and test sets.
- **C-MAPSS Group Splitting**: Partitioning occurs strictly by `engine_id`. Temporal cycles from a single engine never cross partition boundaries.
- **Split Integrity Manifest**: Every run generates a `split-manifest.json` containing disjoint ID sets and an immutable SHA-256 fingerprint.

---

## 4. Signal Representations & Feature Engineering

To support diverse model architectures, FluxGuard AI transforms raw 1D vibration telemetry into three feature domains:

```
Raw Telemetry Window (1D, 2048 Samples)
       │
       ├──► Time-Domain Statistics (10 Statistical Indicators)
       │
       ├──► Frequency-Domain Spectrum (One-Sided Hann-Windowed FFT)
       │
       └──► Time-Frequency Representation:
              ├── Short-Time Fourier Transform (STFT Magnitude Spectrogram)
              └── Continuous Wavelet Transform (CWT Morlet Scalogram)
```

### 4.1 Time-Domain Statistical Indicators
Extracted via `fluxguard.features.time_domain` across 10 guarded indicators:
- **Central Tendency & Dispersion**: Mean, Standard Deviation.
- **Energy Metrics**: Root Mean Square (RMS), Peak-to-Peak Amplitude, Absolute Peak.
- **Distribution Shape**: Skewness, Kurtosis (Fisher definition). High kurtosis highlights impulsive race pitting impacts.
- **Dimensionless Ratios**: Crest Factor ($\frac{\text{Peak}}{\text{RMS}}$), Impulse Factor ($\frac{\text{Peak}}{\text{Mean}(|x|)}$), Form Factor ($\frac{\text{RMS}}{\text{Mean}(|x|)}$).
- *Robustness Guards*: Windows with zero RMS or constant non-zero values are handled gracefully without producing NaNs.

### 4.2 Frequency-Domain Spectra (FFT)
- Applied via `fluxguard.features.frequency_domain`.
- Detrends and centers window samples, applies a Hann window, and executes an $N$-point Discrete Fourier Transform (`np.fft.rfft`).
- Returns one-sided power spectra, frequency bin coordinates, resolution ($\Delta f$), dominant frequency, and custom band energies.

### 4.3 Time-Frequency Image Representations
Transient impacts from bearing ball-pass defects shift frequency over time:
- **STFT (Short-Time Fourier Transform)**: Uses `scipy.signal.stft` with configurable segment size (`n_fft=256`) and hop size (`hop_length=128`) to generate 2D magnitude spectrograms.
- **CWT (Continuous Wavelet Transform)**: Uses `pywt.cwt` with Morlet wavelets (`morl`) across 64 scales to generate 2D scalograms.
- Outputs are resized to compact matrices ($32 \times 32$ or $64 \times 64$) for 2D convolutional networks.

---

## 5. Machine Learning Model Zoo & Architectures

FluxGuard AI employs a tiered model structure combining unsupervised anomaly detection with supervised diagnosis and prognostic regression.

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    subgraph Tier1["Tier 1 - Unsupervised Anomaly Detection"]
        RawInput["Raw Vibration Window - 1x2048"]
        AutoencoderNet["1D Convolutional Autoencoder"]
        ReconstructionLoss["MSE Reconstruction Loss"]
        ThresholdDecider{"Loss > Threshold tau?"}
        NormalStatus["Status - NORMAL"]
        AnomalyTrigger["Status - ANOMALY DETECTED"]
    end

    subgraph Tier2["Tier 2 - Supervised Fault Classification"]
        ClassifierChoice{"Model Variant"}
        Cnn1D["Compact 1D CNN - Raw Waveform"]
        SeTcn["SE-TCN - Dilated Convolutions + SE Attention"]
        SpecCnn["2D CNN - STFT / CWT Images"]
        ClassOutput["Fault Type - Inner / Outer / Ball / Healthy"]
    end

    subgraph Tier3["Tier 3 - Prognostics & RUL Estimation"]
        EngineInput["NASA C-MAPSS Sequence - 30x15 Sensors"]
        LstmNet["Multi-Layer LSTM Regressor"]
        RulOutput["Predicted Remaining Useful Life Cycles"]
    end

    RawInput --> AutoencoderNet
    AutoencoderNet --> ReconstructionLoss
    ReconstructionLoss --> ThresholdDecider
    ThresholdDecider -->|"No"| NormalStatus
    ThresholdDecider -->|"Yes"| AnomalyTrigger
    AnomalyTrigger --> ClassifierChoice
    ClassifierChoice -->|"1D Waveform"| Cnn1D
    ClassifierChoice -->|"Temporal Attn"| SeTcn
    ClassifierChoice -->|"Time-Frequency"| SpecCnn
    Cnn1D --> ClassOutput
    SeTcn --> ClassOutput
    SpecCnn --> ClassOutput
    EngineInput --> LstmNet
    LstmNet --> RulOutput
```

### 5.1 Tier 1: Unsupervised Anomaly Autoencoder (`CWRUAutoencoder`)
- **Objective**: Zero-shot novelty detection without prior fault labels.
- **Architecture**:
  - Encoder: Conv1D (kernel 7, stride 2) $\rightarrow$ ReLU $\rightarrow$ Conv1D (kernel 7, stride 2) $\rightarrow$ ReLU.
  - Bottleneck: 16 latent channels.
  - Decoder: ConvTranspose1D (kernel 4, stride 2) $\rightarrow$ ReLU $\rightarrow$ ConvTranspose1D (kernel 4, stride 2).
- **Training Strategy**: Trained exclusively on healthy operational data (`Normal_0`).
- **Dynamic Threshold Calibration**:
  $$\tau = \mu_{\text{healthy\_val}} + k \cdot \sigma_{\text{healthy\_val}} \quad (\text{default } k = 3.0)$$
  An anomaly is flagged whenever $\mathcal{L}_{\text{MSE}}(x, \hat{x}) > \tau$.

### 5.2 Tier 2: Supervised Fault Classifiers
Classifies physical defect root causes once an anomaly is triggered:
1. **Compact 1D CNN (`CWRUClassifier`)**:
   - Conv1D (kernel 7) $\rightarrow$ MaxPool $\rightarrow$ Conv1D (kernel 5) $\rightarrow$ AdaptiveAvgPool $\rightarrow$ Linear.
   - Lightweight, low latency on CPU.
2. **Squeeze-and-Excitation TCN (`SETCNClassifier`)**:
   - Dilated causal convolutions to capture long-range periodic vibrations without recurrent bottlenecks.
   - Squeeze-and-Excitation (SE) channel gating dynamically weights informative frequency bands.
3. **Spectrogram 2D CNN (`SpectrogramCNN`)**:
   - 2D Conv layers with spatial pooling operating on STFT and CWT matrices.
   - Achieves 95.8% accuracy on benchmark tests by leveraging time-frequency locality.

### 5.3 Tier 3: Remaining Useful Life Regressor (`CMapssRULRegressor`)
- **Objective**: Estimate remaining operational cycles for degraded assets.
- **Architecture**:
  - Input: Sequence tensor $(B, 30, S)$ where $S \in [14, 21]$ active sensors.
  - Recurrent Backbone: Multi-layer LSTM (`hidden_size=64`, `batch_first=True`).
  - Output Head: Linear $\rightarrow$ ReLU $\rightarrow$ Linear(1).
- **Performance**: Achieves an MAE of 9.63 cycles and RMSE of 14.08 cycles on NASA C-MAPSS FD001 (beating linear baselines of 18.40 MAE).

---

## 6. Training Pipeline, Baselines & Quality Gates

FluxGuard AI enforces five automated Quality Gates during experiment execution:

| Gate | Name | Validation Criteria |
|---|---|---|
| Gate 1 | Schema & Provenance Gate | All MAT and C-MAPSS records parsed strictly; invalid files rejected with explicit error types. |
| Gate 2 | Anti-Leakage Gate | CWRU split by recording ID; C-MAPSS split by engine ID; disjoint sets verified with SHA-256 hash. |
| Gate 3 | Non-Deep Baseline Gate | Classical baselines (RMS energy, Logistic Regression, Linear Regression) must run first. |
| Gate 4 | Reproducibility Gate | Fixed seeds produce identical split manifests and deterministic training loss curves. |
| Gate 5 | Artifact Integrity Gate | Checks that checkpoint weights cannot be loaded without accompanying preprocessing metadata. |

### 6.1 Benchmark Model Comparison
Experimental runs executed on the codebase yield the following benchmark performance:

| Experiment / Task | Model Architecture | Primary Metric | Baseline Score | Model Score | Status |
|---|---|---|---|---|---|
| CWRU Anomaly Detection | 1D Conv Autoencoder | ROC-AUC | — | **0.824** | Pass |
| CWRU Anomaly Detection | 1D Conv Autoencoder | FPR / Recall | — | **0.000 / 0.777** | Pass |
| CWRU Fault Classification | 1D CNN (Raw) | Accuracy | 0.958 (Majority) | **0.878** | Pass |
| CWRU Fault Classification | SE-TCN (Raw) | Accuracy / Recall | 0.958 (Majority) | **0.757** (Healthy Rec: 1.0) | Pass |
| CWRU Fault Classification | STFT Spectrogram CNN | Accuracy | 0.958 (Majority) | **0.958** | Pass |
| CWRU Fault Classification | CWT Scalogram CNN | Accuracy | 0.958 (Majority) | **0.958** | Pass |
| NASA C-MAPSS FD001 | LSTM Regressor | MAE (lower better) | 18.399 (Mean) | **9.634** | Pass |

---

## 7. Artifact Management & ONNX Export Pipeline

Every training run produces an immutable, self-contained directory under `models/runs/<experiment>/<run-id>/`:

```
models/runs/<experiment>/<run-id>/
├── model.pt                 # PyTorch state dictionary (source of truth)
├── model.onnx               # Standardized ONNX graph with dynamic batch axes
├── model.onnx.data          # External weight tensor data if large
├── config.json              # Model hyperparams, loss config, and class mappings
├── preprocessing.json       # Scaler means/variances, filter cutoffs, window size
├── split-manifest.json      # Disjoint partition lists and SHA-256 fingerprint
├── metrics.json             # Evaluated metrics alongside non-deep baseline scores
├── onnx.json                # ONNX opset, input/output tensor shapes and axes
├── report.md                # Markdown report with provenance and leakage evidence
└── training-curves.png      # Loss convergence visualization
```

### 7.1 ONNX Export & Edge Quantization Readiness
- Standardized opset version 17/18.
- Dynamic batch axis configured for input and output tensors (`{0: "batch"}`), maintaining fixed sequence and channel dimensions.
- Every exported `.onnx` graph undergoes structural verification via `onnx.checker.check_model` before run sign-off.
- Ready for zero-copy deployment into AMD Quark quantization toolchains and ONNX Runtime edge inference engines.

---

## 8. Inference & Serving Subsystem

FluxGuard AI provides an asynchronous REST API built on FastAPI and Uvicorn.

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    subgraph ClientReq["Client Layer"]
        HttpReq["HTTP REST Client - Gateway / SCADA"]
    end

    subgraph ApiApp["FastAPI Service Application"]
        Router["API Router"]
        HealthRoute["GET /health"]
        AnomalyRoute["POST /api/anomaly/score"]
        FaultRoute["POST /api/fault/predict"]
        RulRoute["POST /api/rul/predict"]
        RegistryLoader["Run Discovery & Model Registry"]
    end

    subgraph MemoryModels["Loaded PyTorch / ONNX Instances"]
        LiveAe["Autoencoder + Threshold tau"]
        LiveClf["Supervised Classifier"]
        LiveRul["LSTM RUL Regressor"]
    end

    HttpReq -->|"HTTP GET"| HealthRoute
    HttpReq -->|"HTTP POST telemetry"| AnomalyRoute
    HttpReq -->|"HTTP POST telemetry"| FaultRoute
    HttpReq -->|"HTTP POST sequences"| RulRoute
    HealthRoute --> Router
    AnomalyRoute -->|"forward array"| LiveAe
    FaultRoute -->|"forward array"| LiveClf
    RulRoute -->|"forward sequence"| LiveRul
    RegistryLoader -->|"injects"| LiveAe
    RegistryLoader -->|"injects"| LiveClf
    RegistryLoader -->|"injects"| LiveRul
```

### 8.1 API Endpoints Specification
- `GET /health`:
  - Returns `{"status": "ok", "models": {"anomaly": bool, "classifier": bool, "rul": bool}}`.
- `POST /api/anomaly/score`:
  - Request: `{"values": [float, ...]}` (vibration window, $\ge 8$ samples).
  - Response: `{"reconstruction_error": 0.0142, "threshold": 0.0208, "anomaly": false}`.
- `POST /api/fault/predict`:
  - Request: `{"values": [float, ...]}`.
  - Response: `{"class_index": 1, "class_name": "inner_race", "probabilities": [0.05, 0.92, 0.02, 0.01]}`.
- `POST /api/rul/predict`:
  - Request: `{"values": [[s1, s2, ...], [s1, s2, ...], ...]}` ($30 \times S$ sensor sequence).
  - Response: `{"predicted_rul": 84.5}`.

### 8.2 Containerized Deployment
A production `Dockerfile` (based on `python:3.12-slim`) packages the application and exposes port 8000:
```powershell
# Build container image
docker build -t fluxguard .

# Run container with mounted trained artifacts
docker run -p 8000:8000 -v ${PWD}/models/runs:/models/runs fluxguard
```

---

## 9. Non-Functional Requirements & Production Readiness

1. **Inference Latency**:
   - Single-window anomaly scoring: $< 15\text{ ms}$ on CPU, $< 2\text{ ms}$ with ONNX Runtime.
   - Sequence RUL prediction: $< 25\text{ ms}$ on CPU.
2. **Air-Gapped Security**:
   - Operates entirely locally without internet access or external SaaS telemetry dependencies.
   - No sensitive industrial operational parameters leave the local network.
3. **Reproducibility**:
   - All random seeds (NumPy, PyTorch, Python random) are centrally managed via `set_seed()`.
   - Every run stores exact data partitions and preprocessing scalers.
4. **Resilience & Error Handling**:
   - Unrecognized file formats raise typed errors (`CWRUDataError`, `CMapssDataError`).
   - Missing or unloaded models in the API gracefully return HTTP 503 instead of crashing.
