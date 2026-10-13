"""Validation helpers for persisted model run bundles."""

import json
from pathlib import Path

REQUIRED_RUN_FILES = (
    "model.pt",
    "config.json",
    "preprocessing.json",
    "split-manifest.json",
    "metrics.json",
)


def validate_run_artifacts(run_dir: Path) -> tuple[bool, tuple[str, ...]]:
    """Return whether a run contains a readable, structurally valid bundle."""
    missing = tuple(name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file())
    if missing:
        return False, tuple(f"missing required artifact: {name}" for name in missing)

    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("kind"):
            return False, ("config.json must contain a non-empty 'kind'",)
        for name in ("preprocessing.json", "split-manifest.json", "metrics.json"):
            value = json.loads((run_dir / name).read_text(encoding="utf-8"))
            if not isinstance(value, (dict, list)):
                return False, (f"{name} must contain a JSON object or array",)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, (f"invalid artifact JSON: {exc}",)

    return True, ()
