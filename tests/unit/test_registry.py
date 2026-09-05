import pytest

from dataleaks.engine.detector import BaseDetector
from dataleaks.engine.registry import DetectorRegistry
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class TestDetector(BaseDetector):
    """Valid detector used for registry tests."""

    name = "test_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        return []


class SecondDetector(BaseDetector):
    """Second valid detector used for registry tests."""

    name = "second_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        return []


class InvalidDetector:
    """Class that does not implement BaseDetector."""

    name = "invalid_detector"


def test_register_valid_detector():
    registry = DetectorRegistry()

    registry.register(TestDetector)

    assert registry.get("test_detector") is TestDetector


def test_register_multiple_detectors():
    registry = DetectorRegistry()

    registry.register(TestDetector)
    registry.register(SecondDetector)

    assert registry.names() == [
        "test_detector",
        "second_detector",
    ]


def test_register_rejects_invalid_detector():
    registry = DetectorRegistry()

    with pytest.raises(TypeError):
        registry.register(InvalidDetector)


def test_register_rejects_duplicate_detector():
    registry = DetectorRegistry()

    registry.register(TestDetector)

    with pytest.raises(ValueError):
        registry.register(TestDetector)


def test_register_rejects_base_detector_name():
    class InvalidBaseDetector(BaseDetector):
        name = "base"
        category = "test"

        def detect(
            self,
            context: DatasetContext,
        ) -> list[Finding]:
            return []

    registry = DetectorRegistry()

    with pytest.raises(ValueError):
        registry.register(InvalidBaseDetector)


def test_get_unknown_detector_raises_key_error():
    registry = DetectorRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_all_returns_registered_detector_classes():
    registry = DetectorRegistry()

    registry.register(TestDetector)
    registry.register(SecondDetector)

    assert registry.all() == [
        TestDetector,
        SecondDetector,
    ]


def test_names_returns_registered_detector_names():
    registry = DetectorRegistry()

    registry.register(TestDetector)
    registry.register(SecondDetector)

    assert registry.names() == [
        "test_detector",
        "second_detector",
    ]


def test_create_all_instantiates_registered_detectors():
    registry = DetectorRegistry()

    registry.register(TestDetector)
    registry.register(SecondDetector)

    detectors = registry.create_all()

    assert len(detectors) == 2
    assert isinstance(detectors[0], TestDetector)
    assert isinstance(detectors[1], SecondDetector)


def test_create_all_returns_empty_list_when_no_detectors():
    registry = DetectorRegistry()

    assert registry.create_all() == []