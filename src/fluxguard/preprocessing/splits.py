"""Deterministic source-grouped split manifests."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitManifest:
    """Train/validation/test groups and a stable configuration fingerprint."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any(not group for group in groups):
            raise ValueError("each split must contain at least one group")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("split groups must be disjoint")

    def as_dict(self) -> dict[str, object]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed": self.seed,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def group_split(
    groups: Iterable[str],
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> SplitManifest:
    """Split unique group IDs deterministically without group leakage."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    unique = sorted({str(group) for group in groups})
    if len(unique) < 3:
        raise ValueError("at least three unique groups are required")
    import random

    random.Random(seed).shuffle(unique)
    train_end = max(1, round(len(unique) * train_fraction))
    validation_end = max(train_end + 1, round(len(unique) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(unique) - 1)
    return SplitManifest(
        tuple(sorted(unique[:train_end])),
        tuple(sorted(unique[train_end:validation_end])),
        tuple(sorted(unique[validation_end:])),
        seed,
    )
