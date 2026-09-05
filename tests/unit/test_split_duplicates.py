import pandas as pd

from dataleaks.detectors.split.duplicates import SplitDuplicateDetector
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


def test_detects_exact_train_test_overlap():
    train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [20000, 30000, 40000],
        }
    )

    test = pd.DataFrame(
        {
            "age": [40, 50],
            "income": [40000, 50000],
        }
    )

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "split_duplicates"
    assert findings[0].category == "split_leakage"
    assert findings[0].evidence["overlap_count"] == 1


def test_clean_splits_produce_no_findings():
    train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [20000, 30000, 40000],
        }
    )

    test = pd.DataFrame(
        {
            "age": [50, 60],
            "income": [50000, 60000],
        }
    )

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert findings == []


def test_duplicate_rows_inside_each_split_are_collapsed():
    train = pd.DataFrame(
        {
            "age": [20, 20, 30],
            "income": [20000, 20000, 30000],
        }
    )

    test = pd.DataFrame(
        {
            "age": [20, 40],
            "income": [20000, 40000],
        }
    )

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1


def test_no_split_information_returns_no_findings():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [20000, 30000, 40000],
        }
    )

    context = make_context(
        train=None,
        test=None,
    )

    context.data = df

    findings = SplitDuplicateDetector().detect(context)

    assert findings == []


def test_empty_train_returns_no_findings():
    train = pd.DataFrame(columns=["age", "income"])

    test = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [20000, 30000],
        }
    )

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert findings == []


def test_empty_test_returns_no_findings():
    train = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [20000, 30000],
        }
    )

    test = pd.DataFrame(columns=["age", "income"])

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert findings == []


def test_different_column_order_is_handled():
    train = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [20000, 30000],
        }
    )

    test = pd.DataFrame(
        {
            "income": [30000, 40000],
            "age": [30, 40],
        }
    )

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1


def test_no_common_columns_returns_no_findings():
    train = pd.DataFrame(
        {
            "age": [20, 30],
        }
    )

    test = pd.DataFrame(
        {
            "salary": [20000, 30000],
        }
    )

    context = make_context(train, test)

    findings = SplitDuplicateDetector().detect(context)

    assert findings == []