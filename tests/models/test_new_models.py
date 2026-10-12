import numpy as np
import pytest
import torch

from fluxguard.models.spectrogram import SpectrogramCNN
from fluxguard.models.tcn import SETCNClassifier


def test_setcn_classifier_accepts_batched_windows() -> None:
    model = SETCNClassifier(class_count=4, input_length=128)
    output = model(torch.randn(3, 128))
    assert output.shape == (3, 4)


def test_setcn_classifier_rejects_bad_input_shape() -> None:
    model = SETCNClassifier(class_count=3, input_length=64)
    with pytest.raises(ValueError):
        model(torch.randn(2, 2, 64))


def test_spectrogram_cnn_accepts_images() -> None:
    model = SpectrogramCNN(class_count=5, height=32, width=32)
    output = model(torch.randn(2, 1, 32, 32))
    assert output.shape == (2, 5)


def test_spectrogram_cnn_accepts_unchannelled_batch() -> None:
    model = SpectrogramCNN(class_count=3, height=32, width=32)
    output = model(torch.randn(4, 32, 32))
    assert output.shape == (4, 3)


def test_generic_trainer_runs_one_epoch() -> None:
    from fluxguard.pipelines.train_classifier_generic import (
        predict_classifier_model,
        train_classifier_model,
    )

    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(16, 64))
    train_y = np.array([0] * 8 + [1] * 8)
    validation_x = rng.normal(size=(8, 64))
    validation_y = np.array([0] * 4 + [1] * 4)
    model, history = train_classifier_model(
        train_x, train_y, validation_x, validation_y,
        model_factory=lambda: SETCNClassifier(2, 64),
        epochs=1, batch_size=8,
    )
    assert len(history.train_loss) == 1
    predictions, probabilities = predict_classifier_model(model, validation_x)
    assert predictions.shape == (8,)
    assert probabilities.shape == (8, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(8), atol=1e-5)
