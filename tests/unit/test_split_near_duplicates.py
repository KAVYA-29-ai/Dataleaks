import pandas as pd
import pytest

from dataleaks.detectors.split.near_duplicates import (
    NearDuplicateDetector,
)
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


def test_detects_near_duplicate_row():
    train = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [50000, 60000],
            "city": ["Delhi", "Agra"],
        }
    )

    test = pd.DataFrame(
        {
            "age": [20],
            "income": [50000],
            "city": ["Mumbai"],
        }
    )

    context = make_context(train, test)

    findings = NearDuplicateDetector(threshold=0.66).detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "split_near_duplicates"
    assert findings[0].category == "split_leakage"


def test_exact_duplicate_is_detected_as_near_duplicate():
    train = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [50000, 60000],
            "city": ["Delhi", "Agra"],
        }
    )

    test = pd.DataFrame(
        {
            "age": [20],
            "income": [50000],
            "city": ["Delhi"],
        }
    )

    context = make_context(train, test)

    findings = NearDuplicateDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].confidence == 1.0


def test_clean_splits_produce_no_findings():
    train = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [50000, 60000],
            "city": ["Delhi", "Agra"],
        }
    )

    test = pd.DataFrame(
        {
            "age": [40],
            "income": [90000],
            "city": ["Pune"],
        }
    )

    context = make_context(train, test)

    findings = NearDuplicateDetector().detect(context)

    assert findings == []


def test_threshold_is_respected():
    train = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
            "c": [3],
            "d": [4],
        }
    )

    test = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
            "c": [3],
            "d": [99],
        }
    )

    context = make_context(train, test)

    assert NearDuplicateDetector(threshold=0.75).detect(context)

    assert NearDuplicateDetector(threshold=1.0).detect(context) == []


def test_missing_values_are_ignored_when_comparing():
    train = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
            "c": [None],
        }
    )

    test = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
            "c": [None],
        }
    )

    context = make_context(train, test)

    findings = NearDuplicateDetector(
        threshold=0.99,
        min_comparable_columns=2,
    ).detect(context)

    assert len(findings) == 1
    assert findings[0].confidence == 1.0

    match = findings[0].evidence["matches"][0]

    assert match["matching_columns"] == 2
    assert match["comparable_columns"] == 2


def test_different_column_order_is_handled():
    train = pd.DataFrame(
        {
            "age": [20],
            "income": [50000],
            "city": ["Delhi"],
        }
    )

    test = pd.DataFrame(
        {
            "city": ["Delhi"],
            "income": [50000],
            "age": [20],
        }
    )

    context = make_context(train, test)

    findings = NearDuplicateDetector().detect(context)

    assert len(findings) == 1


def test_no_split_information_returns_no_findings():
    context = make_context(None, None)

    findings = NearDuplicateDetector().detect(context)

    assert findings == []


def test_empty_split_returns_no_findings():
    train = pd.DataFrame(columns=["a", "b"])
    test = pd.DataFrame(columns=["a", "b"])

    context = make_context(train, test)

    findings = NearDuplicateDetector().detect(context)

    assert findings == []


def test_insufficient_common_columns_returns_no_findings():
    train = pd.DataFrame(
        {
            "a": [1],
        }
    )

    test = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
        }
    )

    context = make_context(train, test)

    findings = NearDuplicateDetector(
        min_comparable_columns=2,
    ).detect(context)

    assert findings == []


def test_invalid_threshold():
    with pytest.raises(ValueError):
        NearDuplicateDetector(threshold=1.5)


def test_invalid_minimum_columns():
    with pytest.raises(ValueError):
        NearDuplicateDetector(min_comparable_columns=0)