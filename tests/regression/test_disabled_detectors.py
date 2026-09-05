import pandas as pd

from dataleaks import DataLeaks
from dataleaks.schemas.config import DataLeaksConfig


def test_disabled_feature_checks_are_explicitly_skipped():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    config = DataLeaksConfig(
        enable_feature_checks=False,
    )

    report = DataLeaks(
        df,
        target="target",
        config=config,
    ).run()

    skipped = {
        execution.detector: execution
        for execution in report.skipped_detectors
    }

    assert "feature_target_encoding" in skipped
    assert "feature_identifier" in skipped
    assert "feature_suspicious" in skipped

    for execution in skipped.values():
        assert execution.status == "skipped"
        assert execution.reason == "disabled_by_config"
        assert execution.finding_count == 0


def test_disabled_feature_checks_do_not_create_findings():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    config = DataLeaksConfig(
        enable_feature_checks=False,
    )

    report = DataLeaks(
        df,
        target="target",
        config=config,
    ).run()

    assert all(
        not finding.detector.startswith("feature_")
        for finding in report.findings
    )


def test_disabled_detectors_are_not_failed():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    config = DataLeaksConfig(
        enable_feature_checks=False,
    )

    report = DataLeaks(
        df,
        target="target",
        config=config,
    ).run()

    assert all(
        execution.status != "failed"
        for execution in report.execution
        if execution.detector.startswith("feature_")
    )


def test_report_exposes_skipped_count():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    config = DataLeaksConfig(
        enable_feature_checks=False,
    )

    report = DataLeaks(
        df,
        target="target",
        config=config,
    ).run()

    assert len(report.skipped_detectors) == 3


def test_disabled_detector_execution_has_no_warning():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    config = DataLeaksConfig(
        enable_feature_checks=False,
    )

    report = DataLeaks(
        df,
        target="target",
        config=config,
    ).run()

    assert report.has_execution_warnings is False