"""End-to-end experiment orchestration: discover → condition → split → train → evaluate → export.

Usage:

    python -m fluxguard.pipelines.run_experiment run --config configs/cwru_anomaly.json
    python -m fluxguard.pipelines.run_experiment summarize

Each run writes a versioned directory under ``models/runs`` containing the
checkpoint, preprocessing metadata, split manifest, metrics (with recorded
baselines), ONNX export, figures, a ``manifest.json`` consumed by the API
loader, and a ``report.md`` with leakage evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fluxguard.config import CMapssConfig, SignalConfig
from fluxguard.datasets.cmapss import (
    add_training_rul,
    fit_sensor_selection,
    make_engine_sequences,
    read_cmapss,
)
from fluxguard.evaluation.metrics import (
    anomaly_metrics,
    classification_metrics,
    regression_metrics,
)
from fluxguard.evaluation.plots import (
    plot_confusion_matrix,
    plot_error_distributions,
    plot_metric_bars,
    plot_prediction_scatter,
    plot_rul_examples,
    plot_threshold_sensitivity,
    plot_training_curves,
)
from fluxguard.evaluation.sensitivity import analyze_threshold_sensitivity
from fluxguard.features.time_frequency import extract_cwt, extract_stft
from fluxguard.models.classifier import CWRUClassifier
from fluxguard.models.spectrogram import SpectrogramCNN
from fluxguard.models.tcn import SETCNClassifier
from fluxguard.pipelines.baselines import (
    feature_logistic_baseline,
    linear_sensor_baseline,
    majority_class_baseline,
    mean_regressor_baseline,
    rms_energy_anomaly_scores,
)
from fluxguard.pipelines.common import (
    append_readme,
    create_run_dir,
    export_onnx,
    save_checkpoint,
    set_seed,
)
from fluxguard.pipelines.train_anomaly import score_autoencoder, train_autoencoder
from fluxguard.pipelines.train_classifier_generic import (
    predict_classifier_model,
    train_classifier_model,
)
from fluxguard.pipelines.train_rul import predict_rul, train_rul
from fluxguard.preprocessing.scaling import fit_training_scaler
from fluxguard.preprocessing.splits import SplitManifest, group_split
from fluxguard.preprocessing.windowing import make_windows

PACKAGE_VERSION = "0.1.0"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration loaded from JSON."""

    name: str
    kind: str
    seed: int
    raw_dir: Path
    output_root: Path
    train_fraction: float
    validation_fraction: float
    training: dict[str, Any]
    model: dict[str, Any]
    signal: dict[str, Any]
    cmapss: dict[str, Any]

    @property
    def signal_config(self) -> SignalConfig:
        return SignalConfig(**self.signal)

    @property
    def cmapss_config(self) -> CMapssConfig:
        return CMapssConfig(**{key: value for key, value in self.cmapss.items() if key != "subset"})


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate one JSON experiment configuration."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("name", "kind", "raw_dir")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"experiment config {path} is missing keys: {missing}")
    kinds = ("cwru-anomaly", "cwru-classifier", "cmapss-rul")
    if payload["kind"] not in kinds:
        raise ValueError(f"experiment kind must be one of {kinds}")
    return ExperimentConfig(
        name=str(payload["name"]),
        kind=str(payload["kind"]),
        seed=int(payload.get("seed", 42)),
        raw_dir=Path(payload["raw_dir"]),
        output_root=Path(payload.get("output_root", "models/runs")),
        train_fraction=float(payload.get("train_fraction", 0.7)),
        validation_fraction=float(payload.get("validation_fraction", 0.15)),
        training=dict(payload.get("training", {})),
        model=dict(payload.get("model", {})),
        signal=dict(payload.get("signal", {})),
        cmapss=dict(payload.get("cmapss", {})),
    )


def _split_evidence(manifest: SplitManifest) -> dict[str, Any]:
    """Build the leakage-evidence block shared by every experiment report."""

    partitions = {"train": manifest.train, "validation": manifest.validation, "test": manifest.test}
    overlap_pairs = [
        f"{left}/{right}"
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        if set(partitions[left]) & set(partitions[right])
    ]
    return {
        "fingerprint": manifest.fingerprint(),
        "seed": manifest.seed,
        "group_counts": {name: len(groups) for name, groups in partitions.items()},
        "disjoint": not overlap_pairs,
        "overlaps": overlap_pairs,
    }


def _provenance(config: ExperimentConfig, manifest: SplitManifest, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "kind": config.kind,
        "seed": config.seed,
        "raw_dir": str(config.raw_dir),
        "split_fingerprint": manifest.fingerprint(),
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": PACKAGE_VERSION,
        **extra,
    }


def _write_report(run_dir: Path, title: str, sections: dict[str, str]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines += [f"## {heading}", "", body.strip(), ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _cap_windows(windows: list[Any], maximum: int) -> list[Any]:
    """Deterministically subsample windows to at most *maximum* items."""

    if maximum <= 0 or len(windows) <= maximum:
        return windows
    indices = np.linspace(0, len(windows) - 1, maximum).round().astype(int)
    return [windows[index] for index in dict.fromkeys(indices)]


def _sensitivity_table(sensitivity: Any) -> str:
    lines = ["| multiplier | healthy FPR | anomaly recall | F1 |", "|---|---|---|---|"]
    for k, fpr, recall, f1 in zip(
        sensitivity.multipliers, sensitivity.false_positive_rates,
        sensitivity.anomaly_recalls, sensitivity.f1_scores,
    ):
        lines.append(f"| {k:.1f} | {fpr:.3f} | {recall:.3f} | {f1:.3f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CWRU anomaly detection
# --------------------------------------------------------------------------------------


def _load_cwru_windows(config: ExperimentConfig) -> tuple[list[Any], dict[str, str]]:
    """Condition CWRU recordings into capped windows plus a recording→label map."""

    from fluxguard.datasets.cwru import CWRUDataError, discover_mat_files, load_record

    signal_config = config.signal_config
    max_per_recording = int(config.model.get("max_windows_per_recording", 12))
    windows: list[Any] = []
    labels: dict[str, str] = {}
    skipped: list[str] = []
    for path in discover_mat_files(config.raw_dir):
        try:
            record, signal = load_record(path, signal_config.sampling_rate_hz)
        except CWRUDataError as exc:
            skipped.append(path.name)
            print(f"warning: skipping unreadable CWRU file: {exc}")
            continue
        # Fault *type* (healthy / outer_race / inner_race / ball / fault) is a
        # stable label; per-file size heuristics are too fine-grained because
        # every CWRU file number would become its own class.
        labels[record.recording_id] = record.fault_type
        recording_windows = make_windows(
            signal, record.recording_id, signal_config, label=record.fault_type
        )
        windows.extend(_cap_windows(recording_windows, max_per_recording))
    if skipped:
        print(f"warning: skipped {len(skipped)} unreadable CWRU files: {', '.join(skipped)}")
    if not labels:
        raise ValueError(f"no readable CWRU recordings found under {config.raw_dir}")
    return windows, labels


def run_cwru_anomaly(config: ExperimentConfig) -> Path:
    """Train the healthy-only autoencoder and record baselines plus sensitivity.

    Healthy recordings are split into train/validation/test groups; every
    fault recording is evaluation-only because the model never trains on
    faults, so this partition assignment cannot leak fault information.
    """

    set_seed(config.seed)
    windows, labels = _load_cwru_windows(config)
    healthy_ids = sorted(recording for recording, label in labels.items() if label == "healthy")
    fault_ids = sorted(recording for recording, label in labels.items() if label != "healthy")
    if len(healthy_ids) < 3:
        raise ValueError("anomaly experiment needs at least three healthy recordings")
    if not fault_ids:
        raise ValueError("anomaly experiment needs at least one fault recording")
    shuffled = healthy_ids.copy()
    random.Random(config.seed).shuffle(shuffled)
    train_count = max(1, len(shuffled) - 2)
    healthy_manifest = SplitManifest(
        train=tuple(sorted(shuffled[:train_count])),
        validation=(shuffled[train_count],),
        test=(shuffled[train_count + 1],),
        seed=config.seed,
    )
    manifest = SplitManifest(
        train=healthy_manifest.train,
        validation=healthy_manifest.validation,
        test=tuple(sorted(healthy_manifest.test + tuple(fault_ids))),
        seed=config.seed,
    )
    partition_of = {
        **{recording: "train" for recording in manifest.train},
        **{recording: "validation" for recording in manifest.validation},
        **{recording: "test" for recording in manifest.test},
    }

    def select_partition(partition: str, healthy_only: bool = False, faults_only: bool = False) -> np.ndarray:
        chosen = [
            window for window in windows
            if partition_of[window.source_id] == partition
            and (not healthy_only or labels[window.source_id] == "healthy")
            and (not faults_only or labels[window.source_id] != "healthy")
        ]
        return np.asarray([window.values for window in chosen], dtype=float)

    train = select_partition("train", healthy_only=True)
    validation = select_partition("validation", healthy_only=True)
    test_healthy = select_partition("test", healthy_only=True)
    test_fault = select_partition("test", faults_only=True)
    if train.size == 0 or validation.size == 0 or test_healthy.size == 0 or test_fault.size == 0:
        raise ValueError(
            "anomaly experiment needs healthy train/validation/test recordings and held-out fault recordings"
        )

    model, threshold, history = train_autoencoder(
        train, validation, seed=config.seed, **config.training
    )

    # Quality gate 3: record a non-deep baseline before the model result.
    baseline_train_scores = rms_energy_anomaly_scores(train)
    baseline_healthy = rms_energy_anomaly_scores(test_healthy)
    baseline_fault = rms_energy_anomaly_scores(test_fault)
    baseline_threshold = float(np.mean(baseline_train_scores) + 3.0 * np.std(baseline_train_scores))
    baseline_scores = anomaly_metrics(
        np.concatenate([np.zeros(len(baseline_healthy)), np.ones(len(baseline_fault))]),
        np.concatenate([baseline_healthy, baseline_fault]),
        baseline_threshold,
    )

    healthy_errors = score_autoencoder(model, test_healthy)
    fault_errors = score_autoencoder(model, test_fault)
    targets = np.concatenate([np.zeros(len(healthy_errors)), np.ones(len(fault_errors))])
    model_scores = anomaly_metrics(
        targets, np.concatenate([healthy_errors, fault_errors]), threshold.threshold
    )
    sensitivity = analyze_threshold_sensitivity(healthy_errors, fault_errors)

    run_dir = create_run_dir(config.output_root, config.name)
    onnx_info = export_onnx(model, torch.randn(2, int(train.shape[1])), run_dir / "model.onnx")
    evidence = _split_evidence(manifest)
    metrics = {
        "model": model_scores,
        "baseline_rms_energy": baseline_scores,
        "threshold": {
            "mean_error": threshold.mean_error,
            "std_error": threshold.std_error,
            "multiplier": threshold.multiplier,
            "value": threshold.threshold,
            "best_multiplier_from_sensitivity": sensitivity.best_multiplier(),
        },
        "sensitivity": sensitivity.as_dict(),
        "counts": {
            "train_windows": len(train),
            "validation_windows": len(validation),
            "test_healthy_windows": len(healthy_errors),
            "test_fault_windows": len(fault_errors),
        },
        "training": {"train_loss": list(history.train_loss), "validation_loss": list(history.validation_loss)},
        "leakage_evidence": evidence,
    }
    preprocessing = {
        "window_size": config.signal_config.window_size,
        "sampling_rate_hz": config.signal_config.sampling_rate_hz,
        "conditioning": "linear detrend + zero-phase Butterworth high-pass",
        "threshold_calibration": "healthy validation windows only",
    }
    artifact_config = {
        "kind": "anomaly",
        "model_class": "CWRUAutoencoder",
        "model_args": {"input_length": int(train.shape[1])},
        "threshold": {
            "mean_error": threshold.mean_error,
            "std_error": threshold.std_error,
            "multiplier": threshold.multiplier,
        },
    }
    save_checkpoint(model, run_dir, artifact_config, preprocessing, manifest.as_dict(), metrics)
    (run_dir / "onnx.json").write_text(json.dumps(onnx_info, indent=2) + "\n", encoding="utf-8")
    append_readme(
        run_dir, f"FluxGuard anomaly run: {config.name}",
        f"Healthy-only CWRU autoencoder. FPR {model_scores['false_positive_rate']:.3f}, "
        f"recall {model_scores['anomaly_recall']:.3f}, ROC-AUC {model_scores.get('roc_auc', float('nan')):.3f}.",
    )
    _write_report(
        run_dir, f"Anomaly detection report: {config.name}",
        {
            "Summary": (
                f"- Model healthy FPR: **{model_scores['false_positive_rate']:.3f}**\n"
                f"- Model fault recall: **{model_scores['anomaly_recall']:.3f}**\n"
                f"- ROC-AUC: **{model_scores.get('roc_auc', float('nan')):.3f}**\n"
                f"- Baseline (RMS energy) FPR: {baseline_scores['false_positive_rate']:.3f}, "
                f"recall {baseline_scores['anomaly_recall']:.3f}\n"
                f"- Threshold: mean + {threshold.multiplier:.1f}·std = {threshold.threshold:.6g}"
            ),
            "Threshold sensitivity": _sensitivity_table(sensitivity),
            "Leakage evidence": (
                f"Split fingerprint `{manifest.fingerprint()[:16]}…`; partitions are disjoint: "
                f"**{evidence['disjoint']}** (train {evidence['group_counts']['train']} / "
                f"validation {evidence['group_counts']['validation']} / "
                f"test {evidence['group_counts']['test']} recordings). "
                "The autoencoder saw healthy training recordings only; threshold calibration "
                "used healthy validation recordings only."
            ),
            "Provenance": "```json\n" + json.dumps(_provenance(config, manifest, {}), indent=2) + "\n```",
        },
    )
    plot_training_curves(history.train_loss, history.validation_loss, run_dir / "training-curves.png")
    plot_error_distributions(healthy_errors, fault_errors, run_dir / "score-distributions.png", threshold.threshold)
    plot_threshold_sensitivity(
        list(sensitivity.multipliers), list(sensitivity.false_positive_rates),
        list(sensitivity.anomaly_recalls), run_dir / "threshold-sensitivity.png",
    )
    plot_metric_bars(
        ["model FPR", "baseline FPR", "model recall", "baseline recall"],
        [
            model_scores["false_positive_rate"], baseline_scores["false_positive_rate"],
            model_scores["anomaly_recall"], baseline_scores["anomaly_recall"],
        ],
        run_dir / "model-vs-baseline.png", "Anomaly detection: model vs RMS baseline",
    )
    return run_dir


# --------------------------------------------------------------------------------------
# CWRU fault classification
# --------------------------------------------------------------------------------------


def _spectrogram_window(window: np.ndarray, sampling_rate_hz: float, representation: str) -> np.ndarray:
    size = (32, 32)
    if representation == "stft":
        return extract_stft(window, sampling_rate_hz, output_size=size).magnitude
    if representation == "cwt":
        return extract_cwt(window, sampling_rate_hz, output_size=size)
    raise ValueError(f"unknown representation: {representation}")


def run_cwru_classifier(config: ExperimentConfig) -> Path:
    """Train a CWRU fault classifier over a chosen representation and architecture."""

    set_seed(config.seed)
    representation = str(config.model.get("representation", "raw"))
    architecture = str(config.model.get("architecture", "cnn"))
    signal_config = config.signal_config
    windows, labels = _load_cwru_windows(config)
    class_names = tuple(sorted(set(labels.values())))
    if len(class_names) < 2:
        raise ValueError("classifier needs at least two fault classes across recordings")
    class_index = {name: index for index, name in enumerate(class_names)}
    manifest = group_split(
        labels.keys(), seed=config.seed,
        train_fraction=config.train_fraction, validation_fraction=config.validation_fraction,
    )

    def select(recordings: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        # _load_cwru_windows already caps windows per recording.
        chosen = [window for window in windows if window.source_id in recordings]
        if representation == "raw":
            inputs = np.asarray([window.values for window in chosen], dtype=float)
        else:
            inputs = np.asarray(
                [
                    _spectrogram_window(window.values, signal_config.sampling_rate_hz, representation)
                    for window in chosen
                ],
                dtype=float,
            )
        targets = np.asarray([class_index[labels[window.source_id]] for window in chosen])
        return inputs, targets

    train_x, train_y = select(manifest.train)
    validation_x, validation_y = select(manifest.validation)
    test_x, test_y = select(manifest.test)
    if train_x.size == 0 or validation_x.size == 0 or test_x.size == 0:
        raise ValueError("classifier experiment needs non-empty train/validation/test partitions")

    baseline_metrics: dict[str, Any] = {
        "majority_class": classification_metrics(test_y, majority_class_baseline(train_y, len(test_y)))
    }
    if representation == "raw":
        baseline_metrics["feature_logistic"] = classification_metrics(
            test_y, feature_logistic_baseline(train_x, train_y, test_x)
        )

    if representation == "raw" and architecture == "cnn":
        def factory() -> torch.nn.Module:
            return CWRUClassifier(len(class_names), int(train_x.shape[1]))

        example = torch.randn(2, int(train_x.shape[1]))
    elif representation == "raw" and architecture == "setcn":
        def factory() -> torch.nn.Module:
            return SETCNClassifier(len(class_names), int(train_x.shape[1]))

        example = torch.randn(2, int(train_x.shape[1]))
    elif representation in ("stft", "cwt"):
        def factory() -> torch.nn.Module:
            return SpectrogramCNN(len(class_names), 32, 32)

        example = torch.randn(2, 1, 32, 32)
    else:
        raise ValueError(
            f"unsupported representation/architecture combination: {representation}/{architecture}"
        )

    model, history = train_classifier_model(
        train_x, train_y, validation_x, validation_y,
        model_factory=factory, seed=config.seed, **config.training,
    )
    predictions, _ = predict_classifier_model(model, test_x)
    scores = classification_metrics(test_y, predictions)

    run_dir = create_run_dir(config.output_root, config.name)
    onnx_info = export_onnx(model, example, run_dir / "model.onnx")
    evidence = _split_evidence(manifest)
    metrics = {
        "model": scores,
        "baselines": baseline_metrics,
        "counts": {
            "train_windows": len(train_y),
            "validation_windows": len(validation_y),
            "test_windows": len(test_y),
            "class_counts_train": {
                name: int(np.sum(train_y == index)) for index, name in enumerate(class_names)
            },
        },
        "training": {"train_loss": list(history.train_loss), "validation_loss": list(history.validation_loss)},
        "leakage_evidence": evidence,
        "representation": representation,
        "architecture": architecture,
    }
    preprocessing = {
        "window_size": signal_config.window_size,
        "sampling_rate_hz": signal_config.sampling_rate_hz,
        "representation": representation,
        "conditioning": "linear detrend + zero-phase Butterworth high-pass",
    }
    artifact_config = {
        "kind": "classifier",
        "model_class": (
            "CWRUClassifier" if representation == "raw" and architecture == "cnn"
            else "SETCNClassifier" if representation == "raw"
            else "SpectrogramCNN"
        ),
        "model_args": (
            {"class_count": len(class_names), "input_length": int(train_x.shape[1])}
            if representation == "raw"
            else {"class_count": len(class_names), "height": 32, "width": 32}
        ),
        "class_names": list(class_names),
        "representation": representation,
        "architecture": architecture,
    }
    save_checkpoint(model, run_dir, artifact_config, preprocessing, manifest.as_dict(), metrics)
    (run_dir / "onnx.json").write_text(json.dumps(onnx_info, indent=2) + "\n", encoding="utf-8")
    append_readme(
        run_dir, f"FluxGuard classifier run: {config.name}",
        f"{architecture}/{representation} CWRU classifier. Accuracy {scores['accuracy']:.3f}, "
        f"macro F1 {scores['macro_f1']:.3f}.",
    )
    best_baseline = max(entry["macro_f1"] for entry in baseline_metrics.values())
    _write_report(
        run_dir, f"Classification report: {config.name}",
        {
            "Summary": (
                f"- Architecture: **{architecture}**, representation: **{representation}**\n"
                f"- Accuracy: **{scores['accuracy']:.3f}**, macro F1: **{scores['macro_f1']:.3f}**, "
                f"weighted F1: {scores['weighted_f1']:.3f}\n"
                f"- Best recorded baseline macro F1: {best_baseline:.3f}\n"
                f"- Classes: {', '.join(class_names)}"
            ),
            "Leakage evidence": (
                f"Split fingerprint `{manifest.fingerprint()[:16]}…`; source recordings are disjoint "
                f"across partitions: **{evidence['disjoint']}**. Windows are never shared between "
                "partitions because the split happens at the recording level before window generation."
            ),
            "Provenance": (
                "```json\n"
                + json.dumps(
                    _provenance(
                        config, manifest, {"representation": representation, "architecture": architecture}
                    ),
                    indent=2,
                )
                + "\n```"
            ),
        },
    )
    plot_training_curves(history.train_loss, history.validation_loss, run_dir / "training-curves.png")
    plot_confusion_matrix(
        np.asarray(scores["confusion_matrix"]), list(class_names), run_dir / "confusion-matrix.png",
        f"{architecture}/{representation} confusion matrix",
    )
    plot_metric_bars(
        list(baseline_metrics) + ["model"],
        [entry["macro_f1"] for entry in baseline_metrics.values()] + [scores["macro_f1"]],
        run_dir / "model-vs-baselines.png", "Macro F1: model vs baselines",
    )
    return run_dir


# --------------------------------------------------------------------------------------
# NASA C-MAPSS RUL
# --------------------------------------------------------------------------------------


def _find_cmapss_file(raw_dir: Path, name: str) -> Path:
    matches = sorted(raw_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"could not find {name} under {raw_dir}")
    return matches[0]


def run_cmapss_rul(config: ExperimentConfig) -> Path:
    """Train the C-MAPSS LSTM RUL regressor with engine-level splits and baselines."""

    set_seed(config.seed)
    subset = str(config.cmapss.get("subset", "FD001"))
    cfg = config.cmapss_config
    frame = add_training_rul(
        read_cmapss(_find_cmapss_file(config.raw_dir, f"train_{subset}.txt")), cfg.rul_cap
    )
    engines = sorted(frame["engine_id"].unique().tolist())
    manifest = group_split(
        [str(engine) for engine in engines], seed=config.seed,
        train_fraction=config.train_fraction, validation_fraction=config.validation_fraction,
    )
    train_ids = {int(engine) for engine in manifest.train}
    validation_ids = {int(engine) for engine in manifest.validation}
    test_ids = {int(engine) for engine in manifest.test}
    train_frame = frame[frame["engine_id"].isin(train_ids)]
    validation_frame = frame[frame["engine_id"].isin(validation_ids)]
    test_frame = frame[frame["engine_id"].isin(test_ids)]

    selection = fit_sensor_selection(train_frame, cfg.variance_threshold)
    selected = list(selection.selected)
    scaler = fit_training_scaler(train_frame[selected].to_numpy(dtype=float), tuple(selected))

    def sequences(part: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled = part.copy()
        scaled[selected] = scaler.transform(part[selected])
        return make_engine_sequences(scaled, selected, cfg.sequence_length, cfg.sequence_stride)

    train_x, train_y, _ = sequences(train_frame)
    validation_x, validation_y, _ = sequences(validation_frame)
    test_x, test_y, _ = sequences(test_frame)
    if train_x.size == 0 or validation_x.size == 0 or test_x.size == 0:
        raise ValueError(
            f"RUL experiment produced empty sequences; engines need >= {cfg.sequence_length} cycles"
        )

    baseline_mean = regression_metrics(test_y, mean_regressor_baseline(train_y, len(test_y)))
    baseline_linear = regression_metrics(
        test_y, linear_sensor_baseline(train_x[:, -1, :], train_y, test_x[:, -1, :])
    )

    model, history = train_rul(
        train_x, train_y, validation_x, validation_y, seed=config.seed, **config.training
    )
    predictions = predict_rul(model, test_x)
    scores = regression_metrics(test_y, predictions)

    run_dir = create_run_dir(config.output_root, config.name)
    onnx_info = export_onnx(model, torch.randn(2, *train_x.shape[1:]), run_dir / "model.onnx")
    evidence = _split_evidence(manifest)
    metrics = {
        "model": scores,
        "baselines": {"mean_rul": baseline_mean, "linear_last_step": baseline_linear},
        "counts": {
            "train_sequences": len(train_y),
            "validation_sequences": len(validation_y),
            "test_sequences": len(test_y),
            "selected_sensors": selected,
            "dropped_sensors": list(selection.dropped),
        },
        "training": {"train_loss": list(history.train_loss), "validation_loss": list(history.validation_loss)},
        "leakage_evidence": evidence,
        "subset": subset,
    }
    preprocessing = {
        "sequence_length": cfg.sequence_length,
        "sequence_stride": cfg.sequence_stride,
        "rul_cap": cfg.rul_cap,
        "scaler": scaler.metadata(),
        "sensor_selection": {
            "selected": list(selection.selected),
            "dropped": list(selection.dropped),
            "threshold": selection.threshold,
            "fitted_on": "training engines only",
        },
    }
    artifact_config = {
        "kind": "rul",
        "model_class": "CMapssRULRegressor",
        "model_args": {"sensor_count": len(selected)},
        "sequence_length": cfg.sequence_length,
        "subset": subset,
    }
    save_checkpoint(model, run_dir, artifact_config, preprocessing, manifest.as_dict(), metrics)
    (run_dir / "onnx.json").write_text(json.dumps(onnx_info, indent=2) + "\n", encoding="utf-8")
    append_readme(
        run_dir, f"FluxGuard RUL run: {config.name} ({subset})",
        f"LSTM RUL regressor on {subset}. MAE {scores['mae']:.3f}, RMSE {scores['rmse']:.3f}; "
        f"mean baseline MAE {baseline_mean['mae']:.3f}.",
    )
    _write_report(
        run_dir, f"RUL report: {config.name} ({subset})",
        {
            "Summary": (
                f"- Subset: **{subset}**\n"
                f"- MAE: **{scores['mae']:.3f}**, RMSE: **{scores['rmse']:.3f}**\n"
                f"- Baselines — mean RUL: MAE {baseline_mean['mae']:.3f}/RMSE {baseline_mean['rmse']:.3f}; "
                f"linear last-step: MAE {baseline_linear['mae']:.3f}/RMSE {baseline_linear['rmse']:.3f}\n"
                f"- Selected sensors ({len(selected)}): {', '.join(selected)}\n"
                f"- Dropped low-variance sensors: {', '.join(selection.dropped) or 'none'}"
            ),
            "Leakage evidence": (
                f"Split fingerprint `{manifest.fingerprint()[:16]}…`; engines are disjoint across "
                f"partitions: **{evidence['disjoint']}**. Sensor variance selection "
                "and the scaler are fitted on training engines only."
            ),
            "Provenance": (
                "```json\n" + json.dumps(_provenance(config, manifest, {"subset": subset}), indent=2) + "\n```"
            ),
        },
    )
    plot_training_curves(history.train_loss, history.validation_loss, run_dir / "training-curves.png")
    plot_prediction_scatter(
        test_y, predictions, run_dir / "prediction-scatter.png", f"{subset} predicted vs actual RUL"
    )
    example_count = min(3, len(validation_ids))
    if example_count:
        cycles: list[list[float]] = []
        actuals: list[list[float]] = []
        predicted_trajectories: list[list[float]] = []
        for engine in sorted(validation_ids)[:example_count]:
            engine_frame = validation_frame[validation_frame["engine_id"] == engine].sort_values("cycle")
            scaled = engine_frame.copy()
            scaled[selected] = scaler.transform(engine_frame[selected])
            values = scaled[selected].to_numpy(dtype=np.float32)
            trajectory: list[float] = []
            for start in range(0, max(0, len(values) - cfg.sequence_length + 1), cfg.sequence_stride):
                window = values[start : start + cfg.sequence_length]
                trajectory.append(float(predict_rul(model, window[None, :, :])[0]))
            if trajectory:
                cycles.append(list(range(cfg.sequence_length, cfg.sequence_length + len(trajectory))))
                actuals.append(engine_frame["rul"].to_numpy(dtype=float)[-len(trajectory):].tolist())
                predicted_trajectories.append(trajectory)
        if cycles:
            plot_rul_examples(cycles, actuals, predicted_trajectories, run_dir / "rul-trajectories.png")
    plot_metric_bars(
        ["model MAE", "baseline MAE", "model RMSE", "baseline RMSE"],
        [scores["mae"], baseline_mean["mae"], scores["rmse"], baseline_mean["rmse"]],
        run_dir / "model-vs-baselines.png", f"{subset}: LSTM vs mean baseline",
    )
    return run_dir


# --------------------------------------------------------------------------------------
# Run summarization and CLI
# --------------------------------------------------------------------------------------


def summarize_runs(root: Path, output: Path) -> Path:
    """Aggregate the latest completed run per experiment into a table and figure."""

    latest: dict[str, Path] = {}
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        latest[metrics_path.parent.parent.name] = metrics_path.parent
    rows: list[dict[str, Any]] = []
    for experiment, run_dir in sorted(latest.items()):
        metrics_path = run_dir / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        kind = str(config_payload.get("kind", "unknown"))
        row: dict[str, Any] = {"run": str(run_dir.relative_to(root)), "kind": kind}
        if kind == "anomaly":
            row["primary"] = payload["model"].get("roc_auc")
            row["metric"] = "roc_auc"
            row["baseline"] = None
        elif kind == "classifier" and "accuracy" in payload.get("model", {}):
            row["primary"] = payload["model"]["accuracy"]
            row["metric"] = "accuracy"
            row["baseline"] = max(
                (entry["accuracy"] for entry in payload.get("baselines", {}).values()), default=None
            )
        elif "mae" in payload.get("model", {}):
            row["primary"] = payload["model"]["mae"]
            row["metric"] = "mae (lower better)"
            row["baseline"] = min(
                (entry["mae"] for entry in payload.get("baselines", {}).values()), default=None
            )
        elif "roc_auc" in payload.get("model", {}):
            row["primary"] = payload["model"]["roc_auc"]
            row["metric"] = "roc_auc"
            row["baseline"] = None
        else:
            continue
        row["disjoint_splits"] = payload.get("leakage_evidence", {}).get("disjoint")
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FluxGuard experiment summary", "",
        "| run | kind | metric | model | baseline | leakage gate |", "|---|---|---|---|---|---|",
    ]
    for row in rows:
        baseline = f"{row['baseline']:.3f}" if row.get("baseline") is not None else "—"
        primary = f"{row['primary']:.3f}" if row.get("primary") is not None else "—"
        lines.append(
            f"| {row['run']} | {row['kind']} | {row['metric']} | {primary} | {baseline} "
            f"| {row['disjoint_splits']} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparable = [row for row in rows if row.get("primary") is not None]
    if comparable:
        plot_metric_bars(
            [row["run"].replace("\\", "/").split("/")[-1] for row in comparable],
            [float(row["primary"]) for row in comparable],
            output.with_suffix(".png"), "Experiment comparison (latest run per experiment)",
        )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FluxGuard experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one experiment from a JSON config")
    run_parser.add_argument("--config", required=True, help="path to an experiment JSON config")
    summarize_parser = subparsers.add_parser("summarize", help="aggregate all completed runs")
    summarize_parser.add_argument("--root", default="models/runs", help="runs root directory")
    summarize_parser.add_argument(
        "--output", default="reports/experiment-summary.md", help="summary markdown path"
    )
    arguments = parser.parse_args(argv)

    if arguments.command == "run":
        config = load_experiment_config(Path(arguments.config))
        runners = {
            "cwru-anomaly": run_cwru_anomaly,
            "cwru-classifier": run_cwru_classifier,
            "cmapss-rul": run_cmapss_rul,
        }
        run_dir = runners[config.kind](config)
        print(f"run complete: {run_dir}")
    else:
        summary = summarize_runs(Path(arguments.root), Path(arguments.output))
        print(f"summary written: {summary}")


if __name__ == "__main__":
    main()
