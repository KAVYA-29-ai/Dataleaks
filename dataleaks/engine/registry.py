from dataleaks.engine.detector import BaseDetector


class DetectorRegistry:
    """Registry for managing DataLeaks detectors."""

    def __init__(self) -> None:
        self._detectors: dict[str, type[BaseDetector]] = {}
        self._skipped: dict[str, str] = {}

    def register(self, detector: type[BaseDetector]) -> None:
        """Register a detector class."""
        if not issubclass(detector, BaseDetector):
            raise TypeError("detector must inherit from BaseDetector")

        name = detector.name

        if not name or name == "base":
            raise ValueError("detector must define a valid name")

        if name in self._detectors:
            raise ValueError(
                f"Detector '{name}' is already registered"
            )

        if name in self._skipped:
            raise ValueError(
                f"Detector '{name}' is already marked as skipped"
            )

        self._detectors[name] = detector

    def skip(
        self,
        detector: type[BaseDetector] | str,
        *,
        reason: str,
    ) -> None:
        """Mark a detector as intentionally skipped."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")

        if isinstance(detector, str):
            name = detector
        else:
            if not issubclass(detector, BaseDetector):
                raise TypeError(
                    "detector must inherit from BaseDetector"
                )
            name = detector.name

        if not name or name == "base":
            raise ValueError("detector must define a valid name")

        if name in self._detectors:
            raise ValueError(
                f"Detector '{name}' is already registered"
            )

        if name in self._skipped:
            raise ValueError(
                f"Detector '{name}' is already marked as skipped"
            )

        self._skipped[name] = reason

    def get(self, name: str) -> type[BaseDetector]:
        """Return a registered detector by name."""
        try:
            return self._detectors[name]
        except KeyError:
            raise KeyError(
                f"Detector '{name}' is not registered"
            ) from None

    def all(self) -> list[type[BaseDetector]]:
        """Return all registered detector classes."""
        return list(self._detectors.values())

    def names(self) -> list[str]:
        """Return registered detector names."""
        return list(self._detectors.keys())

    def skipped(self) -> dict[str, str]:
        """Return detectors intentionally skipped by configuration."""
        return dict(self._skipped)

    def skipped_names(self) -> list[str]:
        """Return skipped detector names in registration order."""
        return list(self._skipped.keys())

    def create_all(self) -> list[BaseDetector]:
        """Instantiate all registered detectors."""
        return [
            detector_class()
            for detector_class in self._detectors.values()
        ]