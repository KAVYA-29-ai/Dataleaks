import pandas as pd

from dataleaks.detectors.temporal.future_features import (
    FutureFeatureDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(
    data,
    *,
    target=None,
    metadata=None,
):
    return DatasetContext(
        data=data,
        target=target,
        metadata=metadata or {},
    )


def test_missing_outcome_date_is_expected_for_negative_target():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
            ],
            "outcome_date": [
                "2025-01-05",
                None,
                None,
            ],
            "churn": [1, 0, 0],
        }
    )

    metadata = {
        "temporal": {
            "conditional_columns": {
                "outcome_date": {
                    "target": "churn",
                    "present_when": 1,
                }
            }
        }
    }

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["outcome_date"],
    )

    findings = detector.detect(
        make_context(
            df,
            target="churn",
            metadata=metadata,
        )
    )

    invalid_findings = [
        finding
        for finding in findings
        if finding.evidence["type"] == "invalid_timestamp"
    ]

    assert invalid_findings == []


def test_missing_outcome_date_is_reported_for_positive_target():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
            ],
            "outcome_date": [
                "2025-01-05",
                None,
                None,
            ],
            "churn": [1, 1, 0],
        }
    )

    metadata = {
        "temporal": {
            "conditional_columns": {
                "outcome_date": {
                    "target": "churn",
                    "present_when": 1,
                }
            }
        }
    }

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["outcome_date"],
    )

    findings = detector.detect(
        make_context(
            df,
            target="churn",
            metadata=metadata,
        )
    )

    invalid_finding = next(
        finding
        for finding in findings
        if finding.evidence["type"] == "invalid_timestamp"
    )

    assert invalid_finding.evidence["invalid_count"] == 1
    assert invalid_finding.evidence[
        "invalid_feature_count"
    ] == 1

    assert invalid_finding.evidence["invalid_rows"] == [
        {
            "row_index": 1,
            "prediction_time": "2025-01-20",
            "feature_time": None,
            "prediction_time_invalid": False,
            "feature_time_invalid": True,
        }
    ]


def test_invalid_outcome_date_is_reported_for_positive_target():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
            ],
            "outcome_date": [
                "2025-01-05",
                "not-a-date",
                None,
            ],
            "churn": [1, 1, 0],
        }
    )

    metadata = {
        "temporal": {
            "conditional_columns": {
                "outcome_date": {
                    "target": "churn",
                    "present_when": 1,
                }
            }
        }
    }

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["outcome_date"],
    )

    findings = detector.detect(
        make_context(
            df,
            target="churn",
            metadata=metadata,
        )
    )

    invalid_finding = next(
        finding
        for finding in findings
        if finding.evidence["type"] == "invalid_timestamp"
    )

    assert invalid_finding.evidence["invalid_count"] == 1
    assert invalid_finding.evidence[
        "invalid_feature_count"
    ] == 1


def test_conditional_missingness_does_not_affect_future_comparison():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
                "2025-01-30",
            ],
            "outcome_date": [
                "2025-01-05",
                None,
                "2025-02-01",
            ],
            "churn": [1, 0, 1],
        }
    )

    metadata = {
        "temporal": {
            "conditional_columns": {
                "outcome_date": {
                    "target": "churn",
                    "present_when": 1,
                }
            }
        }
    }

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["outcome_date"],
    )

    findings = detector.detect(
        make_context(
            df,
            target="churn",
            metadata=metadata,
        )
    )

    future_finding = next(
        finding
        for finding in findings
        if finding.evidence["type"]
        == "future_feature_timestamp"
    )

    assert future_finding.evidence["future_count"] == 1
    assert future_finding.evidence["comparable_count"] == 2


def test_without_conditional_metadata_missing_values_keep_generic_behavior():
    df = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-10",
                "2025-01-20",
            ],
            "outcome_date": [
                "2025-01-05",
                None,
            ],
            "churn": [0, 0],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["outcome_date"],
    )

    findings = detector.detect(
        make_context(
            df,
            target="churn",
        )
    )

    invalid_finding = next(
        finding
        for finding in findings
        if finding.evidence["type"] == "invalid_timestamp"
    )

    assert invalid_finding.evidence["invalid_count"] == 1