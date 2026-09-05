from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataleaks import DataLeaks


def run(
    data: pd.DataFrame,
    target: str = "target",
    **kwargs,
):
    return DataLeaks(
        data,
        target=target,
        **kwargs,
    ).run()


def test_empty_dataframe_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": pd.Series(dtype="float64"),
            "target": pd.Series(dtype="int64"),
        }
    )

    report = run(df)

    assert report is not None
    assert report.findings == []


def test_single_row_dataset_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": [1],
            "target": [0],
        }
    )

    report = run(df)

    assert report is not None


def test_constant_feature_does_not_create_statistical_leakage():
    df = pd.DataFrame(
        {
            "feature": [1, 1, 1, 1, 1],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = run(df)

    statistical = [
        finding
        for finding in report.findings
        if finding.detector
        == "statistical_target_leakage"
    ]

    assert statistical == []


def test_all_null_feature_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": [None, None, None, None],
            "target": [0, 1, 0, 1],
        }
    )

    report = run(df)

    assert report is not None


def test_high_cardinality_non_identifier_is_not_automatically_identifier():
    df = pd.DataFrame(
        {
            "measurement": [
                "a001",
                "a002",
                "a003",
                "a004",
                "a005",
                "a006",
                "a007",
                "a008",
                "a009",
                "a010",
            ],
            "target": [0, 1] * 5,
        }
    )

    report = run(df)

    identifier_findings = [
        finding
        for finding in report.findings
        if finding.detector == "feature_identifier"
        and "measurement"
        in finding.affected_columns
    ]

    assert identifier_findings == []


def test_non_identifier_name_with_unique_values_is_not_identifier():
    df = pd.DataFrame(
        {
            "temperature": range(20),
            "target": [0, 1] * 10,
        }
    )

    report = run(df)

    identifier_findings = [
        finding
        for finding in report.findings
        if finding.detector == "feature_identifier"
    ]

    assert all(
        "temperature"
        not in finding.affected_columns
        for finding in identifier_findings
    )


def test_identifier_name_is_detected_even_when_values_repeat():
    df = pd.DataFrame(
        {
            "customer_id": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            "target": [0, 1] * 4,
        }
    )

    report = run(df)

    identifier_findings = [
        finding
        for finding in report.findings
        if finding.detector == "feature_identifier"
        and "customer_id"
        in finding.affected_columns
    ]

    assert len(identifier_findings) == 1


def test_duplicate_rows_do_not_crash():
    df = pd.DataFrame(
        {
            "feature_a": [1, 1, 1, 2, 2, 3],
            "feature_b": ["a", "a", "a", "b", "b", "c"],
            "target": [0, 0, 0, 1, 1, 0],
        }
    )

    report = run(df)

    assert report is not None


def test_boolean_feature_does_not_crash():
    df = pd.DataFrame(
        {
            "is_active": [
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            "target": [1, 0, 1, 0, 0, 1],
        }
    )

    report = run(df)

    assert report is not None


def test_datetime_feature_does_not_crash():
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "target": [0, 1, 0, 1],
        }
    )

    report = run(df)

    assert report is not None


def test_mixed_numeric_and_string_columns_do_not_crash():
    df = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6],
            "category": ["a", "b", "a", "c", "b", "a"],
            "mixed": [1, "2", 3, "4", None, "6"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    report = run(df)

    assert report is not None


def test_nan_and_inf_values_do_not_crash():
    df = pd.DataFrame(
        {
            "feature": [
                1.0,
                np.nan,
                np.inf,
                -np.inf,
                5.0,
                6.0,
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    report = run(df)

    assert report is not None


def test_missing_target_raises_clear_error():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        (ValueError, KeyError)
    ):
        run(df, target="target")


def test_duplicate_column_names_are_handled():
    df = pd.DataFrame(
        [
            [1, 0],
            [2, 1],
            [3, 0],
        ],
        columns=["feature", "target"],
    )

    report = run(df)

    assert report is not None


def test_string_target_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
            ],
        }
    )

    report = run(df)

    assert report is not None


def test_small_dataset_with_two_classes_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    )

    report = run(df)

    assert report is not None


def test_large_integer_identifier_does_not_crash():
    df = pd.DataFrame(
        {
            "user_id": [
                900000000001,
                900000000002,
                900000000003,
                900000000004,
                900000000005,
            ],
            "feature": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = run(df)

    assert report is not None


def test_identifier_evidence_is_bounded():
    df = pd.DataFrame(
        {
            "customer_id": range(1000),
            "target": [0, 1] * 500,
        }
    )

    report = run(df)

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "feature_identifier"
        and "customer_id"
        in finding.affected_columns
    ]

    assert findings

    evidence = findings[0].evidence

    assert len(
        evidence["row_evidence"]
    ) <= 20


def test_report_execution_is_present():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    report = run(df)

    assert isinstance(
        report.execution,
        list,
    )


def test_report_risk_is_bounded():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = run(df)

    assert 0.0 <= report.risk_score <= 1.0


def test_report_findings_have_valid_severity():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = run(df)

    valid = {
        "low",
        "medium",
        "high",
        "critical",
    }

    assert all(
        finding.severity in valid
        for finding in report.findings
    )


def test_report_confidence_is_bounded():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = run(df)

    assert all(
        0.0 <= finding.confidence <= 1.0
        for finding in report.findings
    )