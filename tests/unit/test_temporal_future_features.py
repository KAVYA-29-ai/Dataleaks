import pandas as pd
import pytest

from dataleaks.detectors.temporal.future_features import (
    FutureFeatureDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(data):
    return DatasetContext(data=data)


def test_detects_future_feature_timestamp():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
            ],
            "last_payment": [
                "2025-01-05",
                "2025-01-25",
                "2025-02-01",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["last_payment"],
    )

    findings = detector.detect(make_context(df))

    assert len(findings) == 1
    assert findings[0].detector == "temporal_future_features"
    assert findings[0].category == "temporal_leakage"
    assert findings[0].evidence["future_count"] == 2


def test_clean_temporal_data_produces_no_findings():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
            ],
            "last_payment": [
                "2025-01-05",
                "2025-01-15",
                "2025-01-25",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["last_payment"],
    )

    findings = detector.detect(make_context(df))

    assert findings == []


def test_equal_timestamp_is_not_future():
    df = pd.DataFrame(
        {
            "prediction_time": ["2025-01-10"],
            "event_time": ["2025-01-10"],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(df))

    assert findings == []


def test_multiple_feature_time_columns():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
            ],
            "payment_time": [
                "2025-01-15",
                "2025-01-15",
            ],
            "signup_time": [
                "2025-01-01",
                "2025-01-05",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=[
            "payment_time",
            "signup_time",
        ],
    )

    findings = detector.detect(make_context(df))

    assert len(findings) == 1
    assert findings[0].affected_columns == [
        "prediction_time",
        "payment_time",
    ]


def test_missing_timestamps_are_reported():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                None,
                "2025-01-30",
            ],
            "event_time": [
                "2025-01-15",
                "2025-02-01",
                None,
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(df))

    assert len(findings) == 2

    invalid_finding = next(
        finding
        for finding in findings
        if finding.evidence["type"] == "invalid_timestamp"
    )

    future_finding = next(
        finding
        for finding in findings
        if finding.evidence["type"]
        == "future_feature_timestamp"
    )

    assert invalid_finding.evidence["invalid_count"] == 2

    assert invalid_finding.evidence[
        "invalid_prediction_count"
    ] == 1

    assert invalid_finding.evidence[
        "invalid_feature_count"
    ] == 1

    assert future_finding.evidence["future_count"] == 1
    assert future_finding.evidence["comparable_count"] == 1


def test_invalid_dates_are_reported():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "invalid",
                "2025-01-10",
            ],
            "event_time": [
                "2025-01-20",
                "invalid",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(df))

    assert len(findings) == 1

    finding = findings[0]

    assert finding.evidence["type"] == (
        "invalid_timestamp"
    )

    assert finding.evidence["invalid_count"] == 2

    assert finding.evidence[
        "invalid_prediction_count"
    ] == 1

    assert finding.evidence[
        "invalid_feature_count"
    ] == 1

    assert finding.evidence["total_rows"] == 2

    assert finding.evidence["invalid_ratio"] == 1.0

    assert len(finding.evidence["invalid_rows"]) == 2


def test_missing_temporal_column_raises_error():
    df = pd.DataFrame(
        {
            "prediction_time": ["2025-01-10"],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["missing_time"],
    )

    with pytest.raises(ValueError):
        detector.detect(make_context(df))


def test_empty_prediction_column_is_rejected():
    with pytest.raises(ValueError):
        FutureFeatureDetector(
            prediction_time_column="",
            feature_time_columns=["event_time"],
        )


def test_empty_feature_columns_are_rejected():
    with pytest.raises(ValueError):
        FutureFeatureDetector(
            prediction_time_column="prediction_time",
            feature_time_columns=[],
        )


def test_severity_scales_with_future_ratio():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
                "2025-02-10",
                "2025-02-20",
            ],
            "event_time": [
                "2025-01-15",
                "2025-01-25",
                "2025-01-25",
                "2025-02-01",
                "2025-02-15",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(df))

    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].confidence == 0.4