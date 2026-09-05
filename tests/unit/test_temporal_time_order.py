import pandas as pd
import pytest

from dataleaks.detectors.temporal.time_order import TimeOrderDetector
from dataleaks.schemas.dataset import DatasetContext


def make_context(train, test):
    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    return DatasetContext(
        data=data,
        train=train,
        test=test,
    )


def test_detects_chronological_violation():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-06-01",
            ],
            "value": [10, 20],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-04-01",
                "2025-05-01",
            ],
            "value": [30, 40],
        }
    )

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "temporal_time_order"
    assert findings[0].category == "temporal_leakage"


def test_correct_chronological_order_produces_no_findings():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-03-01",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-04-01",
                "2025-06-01",
            ],
        }
    )

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_equal_boundary_is_allowed():
    train = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-03-01"],
        }
    )

    test = pd.DataFrame(
        {
            "date": ["2025-03-01", "2025-04-01"],
        }
    )

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_missing_dates_are_ignored():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                None,
                "2025-03-01",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-04-01",
                None,
            ],
        }
    )

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_invalid_dates_are_ignored():
    train = pd.DataFrame(
        {
            "date": [
                "invalid",
                "2025-01-01",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-02-01",
                "invalid",
            ],
        }
    )

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_missing_time_column_in_train_raises_error():
    train = pd.DataFrame(
        {
            "value": [1, 2],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-02-01",
            ],
        }
    )

    context = make_context(train, test)

    with pytest.raises(ValueError):
        TimeOrderDetector(
            time_column="date",
        ).detect(context)


def test_missing_time_column_in_test_raises_error():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-02-01",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "value": [1, 2],
        }
    )

    context = make_context(train, test)

    with pytest.raises(ValueError):
        TimeOrderDetector(
            time_column="date",
        ).detect(context)


def test_no_split_information_returns_no_findings():
    data = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-02-01",
            ],
        }
    )

    context = DatasetContext(data=data)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_empty_splits_return_no_findings():
    train = pd.DataFrame(columns=["date"])
    test = pd.DataFrame(columns=["date"])

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_empty_time_column_is_rejected():
    with pytest.raises(ValueError):
        TimeOrderDetector(time_column="")


def test_evidence_contains_temporal_boundaries():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-06-01",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-04-01",
                "2025-05-01",
            ],
        }
    )

    context = make_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    evidence = findings[0].evidence

    assert evidence["train_max"] == "2025-06-01 00:00:00+00:00"
    assert evidence["test_min"] == "2025-04-01 00:00:00+00:00"