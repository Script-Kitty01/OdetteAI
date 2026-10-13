"""CLI and programmatic utilities for data visualization and signal analysis."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluxguard.config import SignalConfig
from fluxguard.datasets.cmapss import (
    add_training_rul,
    fit_sensor_selection,
    read_cmapss,
)
from fluxguard.datasets.cwru import CWRUDataError, discover_mat_files, load_record
from fluxguard.features.frequency_domain import extract_fft_features
from fluxguard.features.time_domain import extract_time_domain_features
from fluxguard.features.time_frequency import extract_cwt, extract_stft
from fluxguard.preprocessing.windowing import condition_signal, make_windows


def summarize_vibration_data(root: Path) -> pd.DataFrame:
    """Analyze and return summary statistics across discovered CWRU recordings."""
    paths = discover_mat_files(root)
    records_info: list[dict[str, Any]] = []
    config = SignalConfig()

    for path in paths:
        try:
            record, raw_signal = load_record(path, config.sampling_rate_hz)
            clean_signal = condition_signal(raw_signal, config)
            windows = make_windows(raw_signal, record.recording_id, config, record.fault_type)
            stats = extract_time_domain_features(clean_signal[: config.window_size])

            records_info.append(
                {
                    "recording_id": record.recording_id,
                    "fault_type": record.fault_type,
                    "fault_size_in": record.fault_size_in,
                    "load_hp": record.load,
                    "sensor_location": record.sensor_location,
                    "total_samples": len(raw_signal),
                    "window_count": len(windows),
                    "rms": stats["rms"],
                    "kurtosis": stats["kurtosis"],
                    "crest_factor": stats["crest_factor"],
                    "peak_to_peak": stats["peak_to_peak"],
                }
            )
        except (CWRUDataError, OSError, ValueError):
            continue

    return pd.DataFrame(records_info)


def summarize_cmapss_data(path: Path) -> dict[str, Any]:
    """Calculate and return fleet degradation and sensor statistics for C-MAPSS."""
    df = read_cmapss(path)
    df_rul = add_training_rul(df)
    selection = fit_sensor_selection(df_rul)

    engine_lifetimes = df_rul.groupby("engine_id")["cycle"].max()

    sensor_stats: dict[str, dict[str, float]] = {}
    for sensor in selection.selected:
        sensor_stats[sensor] = {
            "mean": float(df_rul[sensor].mean()),
            "std": float(df_rul[sensor].std()),
            "min": float(df_rul[sensor].min()),
            "max": float(df_rul[sensor].max()),
            "variance": float(selection.variances[sensor]),
        }

    return {
        "total_records": len(df_rul),
        "total_engines": int(df_rul["engine_id"].nunique()),
        "selected_sensors": list(selection.selected),
        "dropped_sensors": list(selection.dropped),
        "engine_lifetime_mean": float(engine_lifetimes.mean()),
        "engine_lifetime_min": int(engine_lifetimes.min()),
        "engine_lifetime_max": int(engine_lifetimes.max()),
        "sensor_stats": sensor_stats,
    }


def plot_cwru_multi_domain(
    signal: np.ndarray,
    sampling_rate: float,
    output_path: Path,
    title_prefix: str = "Bearing Vibration Analysis",
) -> dict[str, float]:
    """Plot Time Domain, FFT Spectrum, STFT Spectrogram, and CWT Scalogram in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Compute stats & features
    time_stats = extract_time_domain_features(signal)
    fft_feat = extract_fft_features(signal, sampling_rate)
    stft_feat = extract_stft(signal, sampling_rate, n_fft=256, hop_length=64)
    cwt_scalogram = extract_cwt(signal, sampling_rate, scales=np.arange(1, 65), output_size=(64, 128))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # A: Time domain
    time_axis = np.arange(len(signal)) / sampling_rate
    axes[0, 0].plot(time_axis, signal, color="#1f77b4", lw=0.9)
    axes[0, 0].set_title(f"Time Domain Signal (RMS={time_stats['rms']:.3f}, Kurtosis={time_stats['kurtosis']:.2f})")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude (g)")
    axes[0, 0].grid(alpha=0.3)

    # B: FFT Spectrum
    axes[0, 1].plot(fft_feat.frequencies_hz, fft_feat.magnitude, color="#d62728", lw=1.0)
    axes[0, 1].set_title(f"Frequency Spectrum (Dominant: {fft_feat.dominant_frequency_hz:.1f} Hz)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_xlim(0, sampling_rate / 2)
    axes[0, 1].grid(alpha=0.3)

    # C: STFT Spectrogram
    im1 = axes[1, 0].pcolormesh(
        stft_feat.times_s,
        stft_feat.frequencies_hz,
        stft_feat.magnitude,
        shading="gouraud",
        cmap="viridis",
    )
    axes[1, 0].set_title("STFT Spectrogram")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[1, 0], format="%.2f", pad=0.02)

    # D: CWT Scalogram
    im2 = axes[1, 1].imshow(
        cwt_scalogram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, time_axis[-1], 1, 64],
    )
    axes[1, 1].set_title("CWT Scalogram (Morlet Wavelet)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Scale (1-64)")
    fig.colorbar(im2, ax=axes[1, 1], format="%.2f", pad=0.02)

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return time_stats


def plot_cmapss_engine_trajectories(
    data_path: Path,
    output_path: Path,
    engine_ids: list[int] | None = None,
    sensors_to_plot: list[str] | None = None,
) -> None:
    """Plot multi-sensor degradation trajectories and RUL curves for representative engines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = read_cmapss(data_path)
    df = add_training_rul(df)

    if engine_ids is None:
        engine_ids = [1, 2, 3]

    if sensors_to_plot is None:
        sensors_to_plot = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

    fig, axes = plt.subplots(len(sensors_to_plot) + 1, 1, figsize=(10, 2.5 * (len(sensors_to_plot) + 1)), sharex=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, eng_id in enumerate(engine_ids):
        eng_data = df[df["engine_id"] == eng_id].sort_values("cycle")
        c = colors[i % len(colors)]
        label = f"Engine #{eng_id}"

        # Sensor plots
        for s_idx, sensor in enumerate(sensors_to_plot):
            if sensor in eng_data.columns:
                axes[s_idx].plot(eng_data["cycle"], eng_data[sensor], color=c, label=label, lw=1.2)
                axes[s_idx].set_ylabel(sensor)
                axes[s_idx].grid(alpha=0.3)
                if s_idx == 0:
                    axes[s_idx].set_title("C-MAPSS Sensor Degradation Trajectories Across Cycles", fontweight="bold")

        # RUL plot
        axes[-1].plot(eng_data["cycle"], eng_data["rul"], color=c, label=label, lw=1.5)

    axes[-1].set_ylabel("RUL (Cycles)")
    axes[-1].set_xlabel("Operating Cycle")
    axes[-1].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_class_distributions(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot class distributions and metric distributions (RMS, Kurtosis) by fault category."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Fault class count
    counts = summary_df["fault_type"].value_counts()
    axes[0].bar(counts.index, counts.values, color="#4c78a8", edgecolor="black")
    axes[0].set_title("CWRU Recording Counts by Class")
    axes[0].set_ylabel("Recordings Count")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    # 2. RMS distribution by fault class
    classes = summary_df["fault_type"].unique()
    rms_data = [summary_df[summary_df["fault_type"] == c]["rms"].dropna().values for c in classes]
    axes[1].boxplot(rms_data, tick_labels=classes)
    axes[1].set_title("RMS Energy by Fault Class")
    axes[1].set_ylabel("RMS (g)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.3)

    # 3. Kurtosis distribution by fault class
    kurt_data = [summary_df[summary_df["fault_type"] == c]["kurtosis"].dropna().values for c in classes]
    axes[2].boxplot(kurt_data, tick_labels=classes)
    axes[2].set_title("Kurtosis by Fault Class (Impulsiveness)")
    axes[2].set_ylabel("Kurtosis (Fisher)")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="FluxGuard AI Data Visualizer & Analyzer")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/visualizations"),
        help="Directory to save generated charts and stats tables",
    )
    parser.add_argument(
        "--cwru-dir",
        type=Path,
        default=Path("data/raw/cwru"),
        help="Path to CWRU raw MAT files",
    )
    parser.add_argument(
        "--cmapss-file",
        type=Path,
        default=Path("data/raw/nasa-cmapss/extracted/6. Turbofan Engine Degradation Simulation Data Set/train_FD001.txt"),
        help="Path to C-MAPSS train_FD001.txt file",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FluxGuard AI — Data Visualization and Statistics Pipeline")
    print("=" * 60)

    # 1. CWRU Analysis
    if args.cwru_dir.exists():
        print(f"\n[1/3] Analyzing CWRU vibration recordings in: {args.cwru_dir}")
        cwru_summary = summarize_vibration_data(args.cwru_dir)
        summary_csv = out_dir / "cwru_summary_stats.csv"
        cwru_summary.to_csv(summary_csv, index=False)
        print(f"  -> Discovered {len(cwru_summary)} valid recordings.")
        print(f"  -> Saved summary stats table to: {summary_csv}")

        # Class overview
        plot_class_distributions(cwru_summary, out_dir / "cwru_class_distributions.png")
        print(f"  -> Saved class distribution plot: {out_dir / 'cwru_class_distributions.png'}")

        # Pick one healthy and one fault example to plot multi-domain
        cfg = SignalConfig()
        healthy_paths = [p for p in discover_mat_files(args.cwru_dir) if p.stem.lower() in {"97", "98", "99", "100"}]
        fault_paths = [p for p in discover_mat_files(args.cwru_dir) if p.stem.lower() not in {"97", "98", "99", "100"}]

        if healthy_paths:
            rec, sig = load_record(healthy_paths[0], cfg.sampling_rate_hz)
            clean = condition_signal(sig, cfg)[: cfg.window_size]
            stats = plot_cwru_multi_domain(
                clean,
                cfg.sampling_rate_hz,
                out_dir / "cwru_healthy_multidomain.png",
                f"Healthy Baseline Signal (Record {rec.recording_id})",
            )
            print(f"  -> Generated healthy multi-domain chart: {out_dir / 'cwru_healthy_multidomain.png'}")
            print(f"     [Stats: RMS={stats['rms']:.4f}, Kurtosis={stats['kurtosis']:.2f}, Crest={stats['crest_factor']:.2f}]")

        if fault_paths:
            rec, sig = load_record(fault_paths[0], cfg.sampling_rate_hz)
            clean = condition_signal(sig, cfg)[: cfg.window_size]
            stats = plot_cwru_multi_domain(
                clean,
                cfg.sampling_rate_hz,
                out_dir / "cwru_fault_multidomain.png",
                f"Fault Signal ({rec.fault_type.upper()} - Record {rec.recording_id})",
            )
            print(f"  -> Generated fault multi-domain chart: {out_dir / 'cwru_fault_multidomain.png'}")
            print(f"     [Stats: RMS={stats['rms']:.4f}, Kurtosis={stats['kurtosis']:.2f}, Crest={stats['crest_factor']:.2f}]")
    else:
        print(f"[!] Warning: CWRU directory not found at {args.cwru_dir}")

    # 2. C-MAPSS Analysis
    if args.cmapss_file.exists():
        print(f"\n[2/3] Analyzing NASA C-MAPSS fleet degradation from: {args.cmapss_file.name}")
        cmapss_stats = summarize_cmapss_data(args.cmapss_file)
        print(f"  -> Total Engine Fleet: {cmapss_stats['total_engines']} engines, {cmapss_stats['total_records']} cycle records.")
        print(f"  -> Mean engine run cycles: {cmapss_stats['engine_lifetime_mean']:.1f} (range: {cmapss_stats['engine_lifetime_min']} - {cmapss_stats['engine_lifetime_max']})")
        print(f"  -> Selected sensors (high variance): {len(cmapss_stats['selected_sensors'])}")
        print(f"  -> Dropped sensors (invariant): {len(cmapss_stats['dropped_sensors'])} ({', '.join(cmapss_stats['dropped_sensors'])})")

        plot_cmapss_engine_trajectories(
            args.cmapss_file,
            out_dir / "cmapss_sensor_trajectories.png",
            engine_ids=[1, 2, 3],
        )
        print(f"  -> Generated engine degradation trajectory plot: {out_dir / 'cmapss_sensor_trajectories.png'}")
    else:
        print(f"[!] Warning: C-MAPSS dataset file not found at {args.cmapss_file}")

    print("\n[3/3] Visualization complete! All artifacts saved to:", out_dir.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
