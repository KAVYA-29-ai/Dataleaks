from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from dataleaks import DataLeaks


def run(
    data: pd.DataFrame,
    *,
    target: str = "target",
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
):
    return DataLeaks(
        data,
        target=target,
        train=train,
        test=test,
    ).run()


def split_findings(report):
    return [
        finding
        for finding in report.findings
        if finding.detector == "split_overlap"
    ]


def test_exact_duplicate_rows_between_train_and_test_are_detected():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [4, 5, 6],
            "feature_b": ["d", "e", "f"],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings


def test_completely_disjoint_train_test_rows_are_not_overlap():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [6, 7, 8, 9, 10],
            "feature_b": ["f", "g", "h", "i", "j"],
            "target": [1, 0, 1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings == []


def test_repeated_non_identifier_values_are_not_automatically_overlap():
    train = pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
            "value": [10, 11, 12, 13, 14, 15],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "C",
            ],
            "value": [100, 101, 102],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings == []


def test_unique_identifier_overlap_is_detected():
    train = pd.DataFrame(
        {
            "customer_id": [
                "c001",
                "c002",
                "c003",
                "c004",
                "c005",
            ],
            "feature": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "customer_id": [
                "c004",
                "c005",
                "c006",
            ],
            "feature": [60, 70, 80],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings


def test_unique_non_identifier_feature_is_not_automatically_overlap():
    train = pd.DataFrame(
        {
            "measurement": [
                "m001",
                "m002",
                "m003",
                "m004",
                "m005",
            ],
            "target": [0, 1, 0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "measurement": [
                "m006",
                "m007",
                "m008",
            ],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings == []


def test_near_duplicate_rows_are_detected_when_supported():
    train = pd.DataFrame(
        {
            "feature_a": [100, 200, 300, 400],
            "feature_b": [10, 20, 30, 40],
            "feature_c": ["A", "B", "C", "D"],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [100, 200, 999],
            "feature_b": [10, 20, 999],
            "feature_c": ["A", "B", "Z"],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_nan_values_do_not_crash_split_detector():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, np.nan, 4, 5],
            "feature_b": ["a", "b", "c", None, "e"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [np.nan, 6, 7],
            "feature_b": ["c", None, "g"],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_mixed_dtypes_do_not_crash_split_detector():
    train = pd.DataFrame(
        {
            "integer_feature": [1, 2, 3, 4],
            "float_feature": [1.1, 2.2, 3.3, 4.4],
            "category": ["A", "B", "C", "D"],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "integer_feature": [5, 6],
            "float_feature": [5.5, 6.6],
            "category": ["E", "F"],
            "target": [0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_empty_test_split_does_not_crash():
    train = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [],
            "target": [],
        }
    )

    data = train.copy()

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_empty_train_split_does_not_crash():
    train = pd.DataFrame(
        {
            "feature": [],
            "target": [],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    data = test.copy()

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_single_row_splits_do_not_crash():
    train = pd.DataFrame(
        {
            "feature": [1],
            "target": [0],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [1],
            "target": [1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_duplicate_rows_within_train_only_are_not_train_test_overlap():
    train = pd.DataFrame(
        {
            "feature_a": [1, 1, 1, 2],
            "feature_b": ["A", "A", "A", "B"],
            "target": [0, 0, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [3, 4],
            "feature_b": ["C", "D"],
            "target": [0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings == []


def test_train_test_overlap_evidence_is_bounded():
    train = pd.DataFrame(
        {
            "customer_id": [
                f"customer_{i}"
                for i in range(100)
            ],
            "feature": range(100),
            "target": [0, 1] * 50,
        }
    )

    test = train.iloc[:50].copy()

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings

    for finding in findings:
        evidence = finding.evidence

        for key, value in evidence.items():
            if isinstance(value, list):
                assert len(value) <= 20


def test_split_detector_handles_string_numbers():
    train = pd.DataFrame(
        {
            "feature": [
                "1",
                "2",
                "3",
                "4",
            ],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [
                "5",
                "6",
                "7",
            ],
            "target": [1, 0, 1],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    assert report is not None


def test_completely_identical_train_and_test_are_detected():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": ["A", "B", "C", "D"],
            "target": [0, 1, 0, 1],
        }
    )

    test = train.copy()

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    report = run(
        data,
        train=train,
        test=test,
    )

    findings = split_findings(report)

    assert findings


def test_train_test_without_target_column_raises_validation_error():
    train = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [5, 6, 7],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match=r"Target column 'target' does not exist in the dataset",
    ):
        run(
            data,
            target="target",
            train=train,
            test=test,
        )