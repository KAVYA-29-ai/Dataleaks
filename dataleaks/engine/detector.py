from abc import ABC, abstractmethod

from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class BaseDetector(ABC):
    """Base contract for all DataLeaks detectors."""

    name: str = "base"
    category: str = "unknown"

    @abstractmethod
    def detect(self, context: DatasetContext) -> list[Finding]:
        """Run the detector against the provided dataset context."""
        raise NotImplementedError