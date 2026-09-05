import pandas as pd
import pytest

from dataleaks.detectors.split.overlap import SplitOverlapDetector
from dataleaks.schemas.dataset import DatasetContext


def make_context(train, test):
    if train is None and test is None:
        data = pd.DataFrame()
    else:
        frames = [frame for frame in (train, test) if frame is not None]
        data = pd.concat(frames, ignore_index=True)

    return DatasetContext(
        data=data,
        train=train,
        test=test,
    )


def test_detects_entity_overlap():
    train = pd.DataFrame(
        {
            "user_id": [101, 102, 103],
            "value": [10, 20, 30],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [103, 104],
            "value": [40, 50],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["user_id"]
    ).detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["user_id"]
    assert findings[0].evidence["overlap_count"] == 1


def test_clean_entity_split_produces_no_findings():
    train = pd.DataFrame(
        {
            "user_id": [101, 102],
            "value": [10, 20],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [103, 104],
            "value": [30, 40],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["user_id"]
    ).detect(context)

    assert findings == []


def test_multiple_overlapping_entities_are_detected():
    train = pd.DataFrame(
        {
            "user_id": [101, 102, 103],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [101, 102, 104],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["user_id"]
    ).detect(context)

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 2


def test_multiple_columns_are_checked():
    train = pd.DataFrame(
        {
            "user_id": [101, 102],
            "account_id": [201, 202],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [102, 103],
            "account_id": [203, 202],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["user_id", "account_id"]
    ).detect(context)

    assert len(findings) == 2

    assert {
        finding.affected_columns[0]
        for finding in findings
    } == {"user_id", "account_id"}


def test_missing_values_are_ignored():
    train = pd.DataFrame(
        {
            "user_id": [101, None, 102],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [None, 103],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["user_id"]
    ).detect(context)

    assert findings == []


def test_no_split_information_returns_no_findings():
    context = make_context(None, None)

    findings = SplitOverlapDetector(
        columns=["user_id"]
    ).detect(context)

    assert findings == []


def test_empty_splits_return_no_findings():
    train = pd.DataFrame(columns=["user_id"])
    test = pd.DataFrame(columns=["user_id"])

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["user_id"]
    ).detect(context)

    assert findings == []


def test_missing_configured_column_raises_error():
    train = pd.DataFrame(
        {
            "user_id": [101, 102],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [103, 104],
        }
    )

    context = make_context(train, test)

    with pytest.raises(ValueError):
        SplitOverlapDetector(
            columns=["customer_id"]
        ).detect(context)


def test_empty_columns_configuration_is_rejected():
    with pytest.raises(ValueError):
        SplitOverlapDetector(columns=[])


def test_default_configuration_checks_common_columns():
    train = pd.DataFrame(
        {
            "user_id": [101, 102],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [102, 103],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["user_id"]