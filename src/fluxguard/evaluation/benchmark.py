"""Edge benchmark and PyTorch vs. ONNX parity validation."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fluxguard.models.autoencoder import CWRUAutoencoder
from fluxguard.pipelines.common import load_checkpoint


def benchmark_model_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Measure inference latency and throughput for PyTorch model."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    arr = np.asarray(latencies)
    return {
        "mean_latency_ms": float(np.mean(arr)),
        "std_latency_ms": float(np.std(arr)),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "throughput_fps": float(1000.0 / np.mean(arr)),
    }


def verify_onnx_parity(
    run_dir: Path,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Verify that PyTorch and ONNX models produce numerically identical outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "status": "skipped",
            "parity_pass": None,
            "note": "onnxruntime not installed; skipped runtime evaluation",
            "max_absolute_error": None,
            "tolerance": tolerance,
        }

    pt_path = run_dir / "model.pt"
    onnx_path = run_dir / "model.onnx"
    if not pt_path.is_file() or not onnx_path.is_file():
        raise FileNotFoundError(f"Missing model files in {run_dir}")

    # Load PyTorch model
    ae = CWRUAutoencoder(input_length=2048)
    load_checkpoint(ae, run_dir)
    ae.eval()

    # Load ONNX model
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # Test random sample
    rng = np.random.default_rng(42)
    sample = rng.normal(size=(4, 1, 2048)).astype(np.float32)

    with torch.no_grad():
        pt_out = ae(torch.from_numpy(sample)).cpu().numpy()

    onnx_out = session.run(None, {input_name: sample})[0]

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    parity_pass = bool(max_diff < tolerance)

    return {
        "status": "passed" if parity_pass else "failed",
        "parity_pass": parity_pass,
        "max_absolute_error": max_diff,
        "tolerance": tolerance,
    }
