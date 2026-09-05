import pandas as pd

from dataleaks import DataLeaks


def test_detector_with_zero_findings_is_reported_as_completed():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = DataLeaks(df, target="target").run()

    assert report.execution
    assert report.completed_detectors

    for execution in report.execution:
        if execution.finding_count == 0:
            assert execution.status == "completed"


def test_failed_detector_is_explicitly_recorded():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    dataleaks = DataLeaks(df, target="target")

    original_registry = dataleaks.registry

    class BrokenDetector:
        name = "broken_test_detector"

        def detect(self, context):
            raise RuntimeError("intentional detector failure")

    original_create_all = original_registry.create_all

    def create_all_with_broken_detector():
        detectors = original_create_all()
        detectors.append(BrokenDetector())
        return detectors

    original_registry.create_all = create_all_with_broken_detector

    report = dataleaks.run()

    failed = [
        execution
        for execution in report.execution
        if execution.detector == "broken_test_detector"
    ]

    assert len(failed) == 1
    assert failed[0].status == "failed"
    assert failed[0].error == "intentional detector failure"
    assert failed[0].exception_type == "RuntimeError"
    assert report.has_execution_warnings