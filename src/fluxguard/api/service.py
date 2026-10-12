"""Model registry shared by the inference API and the run loader."""

from fluxguard.models.autoencoder import CWRUAutoencoder
from fluxguard.models.classifier import CWRUClassifier
from fluxguard.models.rul import CMapssRULRegressor


class FluxGuardService:
    """Model registry used by the API and straightforward to replace in deployment."""

    def __init__(
        self,
        anomaly_model: CWRUAutoencoder | None = None,
        anomaly_threshold: float | None = None,
        classifier: CWRUClassifier | None = None,
        class_names: tuple[str, ...] = (),
        rul_model: CMapssRULRegressor | None = None,
    ) -> None:
        self.anomaly_model = anomaly_model
        self.anomaly_threshold = anomaly_threshold
        self.classifier = classifier
        self.class_names = class_names
        self.rul_model = rul_model
