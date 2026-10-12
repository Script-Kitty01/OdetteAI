"""Shared deterministic training, checkpoint, and ONNX artifact utilities."""

import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all supported random generators for reproducible CPU runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "metadata"):
        return value.metadata()
    if hasattr(value, "threshold"):
        return {**asdict(value), "threshold": value.threshold}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def create_run_dir(root: Path, model_name: str, run_id: str | None = None) -> Path:
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / model_name / identifier
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    config: dict[str, Any],
    preprocessing: dict[str, Any],
    split_manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "preprocessing.json", preprocessing)
    write_json(run_dir / "split-manifest.json", split_manifest)
    write_json(run_dir / "metrics.json", metrics)


def load_checkpoint(model: torch.nn.Module, run_dir: Path, device: str = "cpu") -> torch.nn.Module:
    """Load a checkpoint only when its required preprocessing artifacts exist."""

    required = ("model.pt", "config.json", "preprocessing.json", "split-manifest.json", "metrics.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete model artifact {run_dir}; missing {missing}")
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def validate_onnx(path: Path) -> None:
    """Validate an ONNX graph structurally and fail on corrupt exports."""

    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)


def export_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    path: Path,
    input_name: str = "input",
    output_name: str = "output",
) -> dict[str, Any]:
    """Export with the ``torch.export``-based ONNX exporter and a dynamic batch axis.

    Falls back to the legacy TorchScript exporter when the dynamo path cannot
    trace a model, so artifact export never silently disappears from a run.
    """

    model.eval()
    exporter = "dynamo"
    try:
        torch.onnx.export(
            model,
            (example_input,),
            str(path),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_shapes=[{0: "batch"}],
            dynamo=True,
        )
    except (UnicodeEncodeError, ValueError, RuntimeError):
        # The dynamo exporter logs emoji and can crash on Windows consoles
        # using legacy code pages; retry once with a UTF-8 stdout before
        # falling back to the legacy TorchScript exporter.
        import sys

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        try:
            torch.onnx.export(
                model,
                (example_input,),
                str(path),
                input_names=[input_name],
                output_names=[output_name],
                dynamic_shapes=[{0: "batch"}],
                dynamo=True,
            )
        except Exception:  # noqa: BLE001 - legacy fallback is the final safety net
            exporter = "torchscript-legacy"
            torch.onnx.export(
                model,
                (example_input,),
                str(path),
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
    validate_onnx(path)
    return {
        "path": str(path.name),
        "input_names": [input_name],
        "output_names": [output_name],
        "exporter": exporter,
        "dynamic_axes": {input_name: {"0": "batch"}, output_name: {"0": "batch"}},
        "input_shape": list(example_input.shape),
    }


def append_readme(run_dir: Path, title: str, summary: str) -> None:
    (run_dir / "README.md").write_text(f"# {title}\n\n{summary}\n", encoding="utf-8")
