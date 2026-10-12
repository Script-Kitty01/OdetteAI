from pathlib import Path

import torch

from fluxguard.models.autoencoder import CWRUAutoencoder
from fluxguard.pipelines.common import (
    append_readme,
    create_run_dir,
    export_onnx,
    load_checkpoint,
    save_checkpoint,
    validate_onnx,
)


def test_completed_artifact_can_be_loaded_and_exported(tmp_path: Path) -> None:
    model = CWRUAutoencoder(32)
    run_dir = create_run_dir(tmp_path, "autoencoder", "test-run")
    save_checkpoint(model, run_dir, {"seed": 42}, {"window_size": 32}, {"train": ["a"]}, {"loss": 0.1})
    append_readme(run_dir, "Autoencoder", "test artifact")
    loaded = load_checkpoint(CWRUAutoencoder(32), run_dir)
    path = run_dir / "model.onnx"
    export_onnx(loaded, torch.randn(2, 32), path)
    validate_onnx(path)
    assert path.is_file() and (run_dir / "README.md").is_file()
