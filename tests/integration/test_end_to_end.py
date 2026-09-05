import pandas as pd
import pytest

from dataleaks import DataLeaks
from dataleaks.schemas.config import DataLeaksConfig


def test_clean_dataset_produces_no_findings():
    data = pd.DataFrame(
        {
            "age": [21, 25, 31, 42, 29, 35, 48, 26],
            "income": [30000, 45000, 52000, 70000, 41000, 60000, 80000, 39000],
            "target": [0, 1, 0, 1, 0, 1, 1, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    assert report.findings == []
    assert report.has_leakage is False
    assert report.finding_count == 0
    assert report.risk_score == 0.0
    assert report.risk_level == "low"


def test_exact_target_copy_is_detected_end_to_end():
    data = pd.DataFrame(
        {
            "age": [21, 25, 31, 42],
            "leaked_target": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "target_direct" in detectors
    assert report.has_leakage is True
    assert report.risk_score > 0.0
    assert report.risk_level in {
        "high",
        "critical",
    }


def test_identifier_feature_is_detected_end_to_end():
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "age": [21, 25, 31, 42],
            "target": [0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "feature_identifier" in detectors


def test_train_test_duplicate_leakage_is_detected():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [2, 4],
            "target": [1, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        train=train,
        test=test,
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "split_duplicates" in detectors
    assert report.has_leakage is True


def test_temporal_leakage_is_detected_end_to_end():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ],
            "future_feature_time": [
                "2025-01-02",
                "2025-01-01",
                "2025-01-04",
            ],
            "target": [0, 1, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        metadata={
            "temporal": {
                "prediction_time_column": "prediction_time",
                "feature_time_columns": [
                    "future_feature_time"
                ],
            }
        },
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "temporal_future_features" in detectors
    assert report.has_leakage is True


def test_cross_dataset_overlap_is_detected_end_to_end():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    reference = pd.DataFrame(
        {
            "feature": [1, 10],
            "target": [0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        metadata={
            "reference_data": reference,
        },
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "cross_dataset_overlap" in detectors
    assert report.has_leakage is True


def test_disabled_checks_are_not_executed():
    data = pd.DataFrame(
        {
            "leaked_feature": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    config = DataLeaksConfig(
        enable_target_checks=False,
        enable_feature_checks=False,
    )

    report = DataLeaks(
        data,
        target="target",
        config=config,
    ).run()

    detectors = {
        finding.detector
        for finding in report.findings
    }

    assert "target_direct" not in detectors
    assert "target_statistical" not in detectors
    assert "target_derived" not in detectors
    assert "feature_suspicious" not in detectors
    assert "feature_identifier" not in detectors
    assert "feature_target_encoding" not in detectors


def test_complete_report_contains_all_expected_sections():
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "leaked_feature": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    assert isinstance(report.findings, list)
    assert isinstance(report.risk_score, float)
    assert isinstance(report.risk_level, str)
    assert isinstance(report.average_confidence, float)
    assert isinstance(report.highest_confidence, float)
    assert isinstance(report.recommendations, list)
    assert isinstance(report.metadata, dict)