"""Training-only scaling with serializable metadata."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedScaler:
    """A StandardScaler fitted on training data only."""

    scaler: StandardScaler
    columns: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(values)

    def metadata(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
        }


def fit_training_scaler(values: np.ndarray, columns: tuple[str, ...]) -> FittedScaler:
    """Fit a scaler and reject empty or non-finite training input."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(columns) or not np.isfinite(array).all():
        raise ValueError("training values must be a finite 2D array matching columns")
    if array.shape[0] == 0:
        raise ValueError("training values cannot be empty")
    return FittedScaler(StandardScaler().fit(array), columns)
