import pandas as pd
import pytest

from dataleaks.engine.detector import BaseDetector
from dataleaks.engine.registry import DetectorRegistry
from dataleaks.engine.runner import DetectorRunner
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


def make_context():
    return DatasetContext(
        data=pd.DataFrame(
            {
                "feature": [1, 2, 3],
                "target": [0, 1, 0],
            }
        ),
        target="target",
    )


class EmptyDetector(BaseDetector):
    name = "empty_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        return []


class SecondEmptyDetector(BaseDetector):
    name = "second_empty_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        return []


class FindingDetector(BaseDetector):
    name = "finding_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity="high",
                confidence=0.9,
                explanation="Test finding",
                recommendation="Test recommendation",
                affected_columns=["feature"],
                evidence={"type": "test"},
            )
        ]


class SecondFindingDetector(BaseDetector):
    name = "second_finding_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity="medium",
                confidence=0.8,
                explanation="Second test finding",
                recommendation="Second test recommendation",
                affected_columns=["feature"],
                evidence={"type": "test"},
            )
        ]


class FailingDetector(BaseDetector):
    name = "failing_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        raise RuntimeError("simulated detector failure")


class InvalidResultDetector(BaseDetector):
    name = "invalid_result_detector"
    category = "test"

    def detect(self, context: DatasetContext):
        return None


class ContextCheckingDetector(BaseDetector):
    name = "context_checking_detector"
    category = "test"

    received_context = None

    def detect(self, context: DatasetContext) -> list[Finding]:
        ContextCheckingDetector.received_context = context
        return []


def test_runner_returns_empty_list_when_no_detectors():
    registry = DetectorRegistry()
    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert findings == []


def test_runner_collects_findings_from_detector():
    registry = DetectorRegistry()
    registry.register(FindingDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert len(findings) == 1
    assert findings[0].detector == "finding_detector"
    assert findings[0].severity == "high"


def test_runner_collects_findings_from_multiple_detectors():
    registry = DetectorRegistry()

    registry.register(FindingDetector)
    registry.register(SecondFindingDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert len(findings) == 2
    assert [finding.detector for finding in findings] == [
        "finding_detector",
        "second_finding_detector",
    ]


def test_runner_preserves_detector_registration_order():
    registry = DetectorRegistry()

    registry.register(FindingDetector)
    registry.register(SecondFindingDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert [finding.detector for finding in findings] == [
        "finding_detector",
        "second_finding_detector",
    ]


def test_runner_preserves_order_across_repeated_runs():
    registry = DetectorRegistry()

    registry.register(FindingDetector)
    registry.register(SecondFindingDetector)

    runner = DetectorRunner(registry)
    context = make_context()

    first = runner.run(context)
    second = runner.run(context)

    assert [finding.detector for finding in first] == [
        "finding_detector",
        "second_finding_detector",
    ]

    assert [finding.detector for finding in second] == [
        "finding_detector",
        "second_finding_detector",
    ]


def test_empty_detector_result_does_not_create_warning():
    registry = DetectorRegistry()
    registry.register(EmptyDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert findings == []
    assert runner.warnings == []


def test_detector_failure_does_not_stop_pipeline():
    registry = DetectorRegistry()

    registry.register(FailingDetector)
    registry.register(FindingDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert len(findings) == 1
    assert findings[0].detector == "finding_detector"


def test_detector_failure_is_recorded_as_warning():
    registry = DetectorRegistry()
    registry.register(FailingDetector)

    runner = DetectorRunner(registry)

    runner.run(make_context())

    assert len(runner.warnings) == 1
    assert runner.warnings[0]["detector"] == "failing_detector"
    assert runner.warnings[0]["error"] == "simulated detector failure"


def test_detector_failure_warning_contains_exception_type():
    registry = DetectorRegistry()
    registry.register(FailingDetector)

    runner = DetectorRunner(registry)

    runner.run(make_context())

    assert runner.warnings[0]["exception_type"] == "RuntimeError"


def test_detector_failure_does_not_hide_successful_findings():
    registry = DetectorRegistry()

    registry.register(FindingDetector)
    registry.register(FailingDetector)
    registry.register(SecondFindingDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert len(findings) == 2
    assert [finding.detector for finding in findings] == [
        "finding_detector",
        "second_finding_detector",
    ]

    assert len(runner.warnings) == 1
    assert runner.warnings[0]["detector"] == "failing_detector"


def test_successful_detectors_produce_no_warnings():
    registry = DetectorRegistry()

    registry.register(EmptyDetector)
    registry.register(FindingDetector)

    runner = DetectorRunner(registry)

    runner.run(make_context())

    assert runner.warnings == []


def test_runner_warning_state_is_reset_between_runs():
    registry = DetectorRegistry()
    registry.register(FailingDetector)

    runner = DetectorRunner(registry)

    runner.run(make_context())

    assert len(runner.warnings) == 1

    runner.run(make_context())

    assert len(runner.warnings) == 1


def test_invalid_detector_result_raises_type_error():
    registry = DetectorRegistry()
    registry.register(InvalidResultDetector)

    runner = DetectorRunner(registry)

    with pytest.raises(
        TypeError,
        match="must return list",
    ):
        runner.run(make_context())


def test_invalid_detector_result_does_not_create_warning():
    registry = DetectorRegistry()
    registry.register(InvalidResultDetector)

    runner = DetectorRunner(registry)

    with pytest.raises(TypeError):
        runner.run(make_context())

    assert runner.warnings == []


def test_runner_passes_same_context_to_detector():
    context = make_context()

    registry = DetectorRegistry()
    registry.register(ContextCheckingDetector)

    runner = DetectorRunner(registry)

    runner.run(context)

    assert ContextCheckingDetector.received_context is context


def test_detector_failure_warning_is_specific_to_detector():
    registry = DetectorRegistry()

    class FirstFailingDetector(BaseDetector):
        name = "first_failing_detector"
        category = "test"

        def detect(self, context: DatasetContext) -> list[Finding]:
            raise ValueError("first failure")

    class SecondFailingDetector(BaseDetector):
        name = "second_failing_detector"
        category = "test"

        def detect(self, context: DatasetContext) -> list[Finding]:
            raise RuntimeError("second failure")

    registry.register(FirstFailingDetector)
    registry.register(SecondFailingDetector)

    runner = DetectorRunner(registry)

    findings = runner.run(make_context())

    assert findings == []
    assert len(runner.warnings) == 2

    assert runner.warnings[0]["detector"] == "first_failing_detector"
    assert runner.warnings[0]["error"] == "first failure"
    assert runner.warnings[0]["exception_type"] == "ValueError"

    assert runner.warnings[1]["detector"] == "second_failing_detector"
    assert runner.warnings[1]["error"] == "second failure"
    assert runner.warnings[1]["exception_type"] == "RuntimeError"