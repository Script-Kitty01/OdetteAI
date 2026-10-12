import numpy as np
import torch

from fluxguard.models.autoencoder import CWRUAutoencoder, calibrate_threshold
from fluxguard.models.classifier import CWRUClassifier
from fluxguard.models.rul import CMapssRULRegressor
from fluxguard.pipelines.train_anomaly import train_autoencoder
from fluxguard.pipelines.train_classifier import train_classifier
from fluxguard.pipelines.train_rul import train_rul


def test_model_shapes_and_threshold() -> None:
    assert CWRUAutoencoder(32)(torch.randn(2, 32)).shape == (2, 1, 32)
    assert CWRUClassifier(3, 32)(torch.randn(2, 32)).shape == (2, 3)
    assert CMapssRULRegressor(4, 8)(torch.randn(2, 5, 4)).shape == (2,)
    threshold = calibrate_threshold(np.array([1.0, 2.0, 3.0]))
    assert threshold.threshold > 0


def test_training_helpers_run_cpu() -> None:
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(8, 32)).astype(np.float32)
    _, threshold, history = train_autoencoder(windows[:6], windows[6:], epochs=1, batch_size=2)
    assert threshold.threshold >= 0
    assert len(history.train_loss) == 1

    labels = np.array([0, 1, 0, 1, 0, 1])
    _, classifier_history = train_classifier(
        windows[:6], labels, windows[6:], np.array([0, 1]), class_count=2, epochs=1, batch_size=2
    )
    assert len(classifier_history.validation_loss) == 1

    sequences = rng.normal(size=(8, 5, 4)).astype(np.float32)
    targets = rng.uniform(size=8).astype(np.float32)
    _, rul_history = train_rul(
        sequences[:6], targets[:6], sequences[6:], targets[6:], epochs=1, batch_size=2
    )
    assert len(rul_history.train_loss) == 1
