import numpy as np
import pandas as pd
import pytest

from dataleaks import DataLeaks


def test_empty_dataframe_does_not_crash():
    df = pd.DataFrame(columns=["feature_a", "feature_b", "target"])

    report = DataLeaks(df, target="target").run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_single_row_dataframe_does_not_crash():
    df = pd.DataFrame(
        {
            "feature_a": [10],
            "feature_b": [20],
            "target": [1],
        }
    )

    report = DataLeaks(df, target="target").run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_missing_target_column_fails_explicitly():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
        }
    )

    with pytest.raises((ValueError, KeyError)):
        DataLeaks(df, target="target").run()


def test_target_column_typo_fails_explicitly():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    with pytest.raises((ValueError, KeyError)):
        DataLeaks(df, target="targte").run()


def test_constant_target_does_not_crash():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [5, 4, 3, 2, 1],
            "target": [1, 1, 1, 1, 1],
        }
    )

    report = DataLeaks(df, target="target").run()

    assert report is not None
    assert isinstance(report.findings, list)

    for finding in report.findings:
        assert not np.isnan(finding.confidence)
        assert 0.0 <= finding.confidence <= 1.0


def test_all_nan_column_does_not_crash():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = DataLeaks(df, target="target").run()

    assert report is not None
    assert isinstance(report.findings, list)

    for finding in report.findings:
        assert not np.isnan(finding.confidence)
        assert 0.0 <= finding.confidence <= 1.0


def test_non_numeric_target_does_not_crash():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [5, 4, 3, 2, 1],
            "target": ["yes", "no", "yes", "no", "yes"],
        }
    )

    report = DataLeaks(df, target="target").run()

    assert report is not None
    assert isinstance(report.findings, list)


def test_duplicate_column_names_do_not_silently_produce_invalid_report():
    df = pd.DataFrame(
        [
            [1, 10, 0],
            [2, 20, 1],
            [3, 30, 0],
        ],
        columns=["feature", "feature", "target"],
    )

    try:
        report = DataLeaks(df, target="target").run()
    except (ValueError, KeyError):
        return

    assert report is not None
    assert isinstance(report.findings, list)

    for finding in report.findings:
        assert finding.detector
        assert finding.severity
        assert 0.0 <= finding.confidence <= 1.0