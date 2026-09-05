import pandas as pd
import pytest

from dataleaks import DataLeaks


def test_test_dataset_with_extra_columns_does_not_crash():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [5, 6],
            "feature_b": [50, 60],
            "extra_feature": [100, 200],
            "target": [1, 0],
        }
    )

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_test_dataset_with_missing_columns_does_not_crash():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [5, 6],
            "target": [1, 0],
        }
    )

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_train_and_test_target_dtype_mismatch_does_not_crash():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [5, 6],
            "target": ["1", "0"],
        }
    )

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_empty_test_dataset_does_not_crash():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": pd.Series(dtype="int64"),
            "target": pd.Series(dtype="int64"),
        }
    )

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_identical_train_and_test_are_reported_as_overlap():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    test = train.copy()

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    assert report is not None

    split_findings = [
        finding
        for finding in report.findings
        if finding.detector.startswith("split")
    ]

    assert split_findings

    assert any(
        finding.severity in {"high", "critical"}
        for finding in split_findings
    )


def test_identical_train_and_test_report_has_overlap_evidence():
    train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    test = train.copy()

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    overlap_findings = [
        finding
        for finding in report.findings
        if finding.detector in {
            "split_duplicates",
            "split_overlap",
            "split_near_duplicates",
        }
    ]

    assert overlap_findings

    for finding in overlap_findings:
        assert finding.evidence