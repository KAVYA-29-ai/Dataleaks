import json

import pandas as pd
import pytest

from dataleaks import DataLeaks
from dataleaks.engine.detector import BaseDetector
from dataleaks.engine.registry import DetectorRegistry
from dataleaks.engine.runner import DetectorRunner
from dataleaks.reporting.json import JSONReporter
from dataleaks.reporting.report import LeakageReport
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.execution import DetectorExecution
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


class CleanDetector(BaseDetector):
    name = "clean_detector"
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
                explanation="Test finding.",
                recommendation="Fix test finding.",
            )
        ]


class FailingDetector(BaseDetector):
    name = "failing_detector"
    category = "test"

    def detect(self, context: DatasetContext) -> list[Finding]:
        raise RuntimeError("execution failure")


def make_registry(*detectors):
    registry = DetectorRegistry()

    for detector in detectors:
        registry.register(detector)

    return registry


def test_clean_detector_is_recorded_as_completed_with_zero_findings():
    runner = DetectorRunner(
        make_registry(CleanDetector)
    )

    findings = runner.run(make_context())

    assert findings == []
    assert len(runner.execution) == 1

    execution = runner.execution[0]

    assert execution.detector == "clean_detector"
    assert execution.status == "completed"
    assert execution.finding_count == 0
    assert execution.error is None
    assert execution.exception_type is None


def test_detector_with_findings_is_recorded_as_completed():
    runner = DetectorRunner(
        make_registry(FindingDetector)
    )

    findings = runner.run(make_context())

    assert len(findings) == 1
    assert len(runner.execution) == 1

    execution = runner.execution[0]

    assert execution.detector == "finding_detector"
    assert execution.status == "completed"
    assert execution.finding_count == 1
    assert execution.error is None
    assert execution.exception_type is None


def test_failed_detector_is_recorded_as_failed():
    runner = DetectorRunner(
        make_registry(FailingDetector)
    )

    findings = runner.run(make_context())

    assert findings == []
    assert len(runner.execution) == 1

    execution = runner.execution[0]

    assert execution.detector == "failing_detector"
    assert execution.status == "failed"
    assert execution.finding_count == 0
    assert execution.error == "execution failure"
    assert execution.exception_type == "RuntimeError"


def test_failed_detector_is_also_recorded_as_warning():
    runner = DetectorRunner(
        make_registry(FailingDetector)
    )

    runner.run(make_context())

    assert len(runner.warnings) == 1
    assert runner.warnings[0]["detector"] == "failing_detector"
    assert runner.warnings[0]["error"] == "execution failure"
    assert runner.warnings[0]["exception_type"] == "RuntimeError"


def test_completed_and_failed_detectors_are_distinguishable():
    runner = DetectorRunner(
        make_registry(
            CleanDetector,
            FailingDetector,
            FindingDetector,
        )
    )

    findings = runner.run(make_context())

    assert len(findings) == 1
    assert len(runner.execution) == 3

    assert [
        execution.status
        for execution in runner.execution
    ] == [
        "completed",
        "failed",
        "completed",
    ]

    assert [
        execution.detector
        for execution in runner.execution
    ] == [
        "clean_detector",
        "failing_detector",
        "finding_detector",
    ]


def test_report_preserves_execution_information():
    execution = [
        DetectorExecution(
            detector="clean_detector",
            status="completed",
            finding_count=0,
        ),
        DetectorExecution(
            detector="failing_detector",
            status="failed",
            finding_count=0,
            error="execution failure",
            exception_type="RuntimeError",
        ),
    ]

    report = LeakageReport.from_findings(
        [],
        execution=execution,
    )

    assert len(report.execution) == 2
    assert report.execution == execution


def test_report_exposes_completed_detectors():
    execution = [
        DetectorExecution(
            detector="clean_detector",
            status="completed",
            finding_count=0,
        ),
        DetectorExecution(
            detector="failing_detector",
            status="failed",
            error="failure",
            exception_type="RuntimeError",
        ),
    ]

    report = LeakageReport.from_findings(
        [],
        execution=execution,
    )

    assert [
        item.detector
        for item in report.completed_detectors
    ] == ["clean_detector"]


def test_report_exposes_failed_detectors():
    execution = [
        DetectorExecution(
            detector="clean_detector",
            status="completed",
        ),
        DetectorExecution(
            detector="failing_detector",
            status="failed",
            error="failure",
            exception_type="RuntimeError",
        ),
    ]

    report = LeakageReport.from_findings(
        [],
        execution=execution,
    )

    assert [
        item.detector
        for item in report.failed_detectors
    ] == ["failing_detector"]

    assert report.has_execution_warnings is True


def test_report_without_execution_has_no_execution_warnings():
    report = LeakageReport.from_findings([])

    assert report.execution == []
    assert report.completed_detectors == []
    assert report.failed_detectors == []
    assert report.has_execution_warnings is False


def test_api_propagates_execution_to_report():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        registry=make_registry(
            CleanDetector,
            FindingDetector,
        ),
    ).run()

    assert len(report.execution) == 2

    assert [
        execution.detector
        for execution in report.execution
    ] == [
        "clean_detector",
        "finding_detector",
    ]

    assert [
        execution.status
        for execution in report.execution
    ] == [
        "completed",
        "completed",
    ]

    assert [
        execution.finding_count
        for execution in report.execution
    ] == [
        0,
        1,
    ]


def test_api_propagates_detector_failure_to_report():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        registry=make_registry(
            CleanDetector,
            FailingDetector,
        ),
    ).run()

    assert len(report.execution) == 2

    assert report.execution[0].status == "completed"
    assert report.execution[0].finding_count == 0

    assert report.execution[1].status == "failed"
    assert report.execution[1].detector == "failing_detector"
    assert report.execution[1].error == "execution failure"
    assert report.execution[1].exception_type == "RuntimeError"

    assert report.has_execution_warnings is True


def test_json_report_contains_execution_information():
    execution = [
        DetectorExecution(
            detector="clean_detector",
            status="completed",
            finding_count=0,
        ),
        DetectorExecution(
            detector="failing_detector",
            status="failed",
            error="execution failure",
            exception_type="RuntimeError",
        ),
    ]

    report = LeakageReport.from_findings(
        [],
        execution=execution,
    )

    data = JSONReporter().to_dict(report)

    assert "execution" in data
    assert data["execution"]["completed_count"] == 1
    assert data["execution"]["failed_count"] == 1
    assert data["execution"]["has_warnings"] is True

    assert data["execution"]["detectors"] == [
        {
            "detector": "clean_detector",
            "status": "completed",
            "finding_count": 0,
            "error": None,
            "exception_type": None,
        },
        {
            "detector": "failing_detector",
            "status": "failed",
            "finding_count": 0,
            "error": "execution failure",
            "exception_type": "RuntimeError",
        },
    ]


def test_json_execution_information_is_serializable():
    execution = [
        DetectorExecution(
            detector="clean_detector",
            status="completed",
            finding_count=0,
        ),
    ]

    report = LeakageReport.from_findings(
        [],
        execution=execution,
    )

    rendered = JSONReporter().render(report)

    parsed = json.loads(rendered)

    assert parsed["execution"]["completed_count"] == 1
    assert parsed["execution"]["failed_count"] == 0
    assert parsed["execution"]["has_warnings"] is False


def test_execution_order_is_deterministic():
    registry = make_registry(
        CleanDetector,
        FailingDetector,
        FindingDetector,
    )

    runner = DetectorRunner(registry)
    context = make_context()

    runner.run(context)
    first_order = [
        execution.detector
        for execution in runner.execution
    ]

    runner.run(context)
    second_order = [
        execution.detector
        for execution in runner.execution
    ]

    assert first_order == second_order
    assert first_order == [
        "clean_detector",
        "failing_detector",
        "finding_detector",
    ]


def test_execution_schema_rejects_invalid_status():
    try:
        DetectorExecution(
            detector="test",
            status="skipped",
        )
    except ValueError as exc:
        assert "status must be" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_failed_execution_requires_error_and_exception_type():
    with pytest.raises(ValueError):
        DetectorExecution(
            detector="test",
            status="failed",
        )


def test_completed_execution_cannot_have_error():
    try:
        DetectorExecution(
            detector="test",
            status="completed",
            error="unexpected error",
        )
    except ValueError as exc:
        assert "completed execution cannot contain an error" in str(exc)
    else:
        raise AssertionError("Expected ValueError")