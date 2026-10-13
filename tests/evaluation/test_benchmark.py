from pathlib import Path

from fluxguard.evaluation.benchmark import verify_onnx_parity


def test_onnx_parity_missing_runtime_is_not_a_pass(tmp_path: Path) -> None:
    result = verify_onnx_parity(tmp_path)

    assert result["status"] == "skipped"
    assert result["parity_pass"] is None
