import pandas as pd

from dataleaks.detectors.split.overlap import SplitOverlapDetector
from dataleaks.schemas.dataset import DatasetContext


def make_context(train, test, target=None):
    return DatasetContext(
        data=train,
        target=target,
        train=train,
        test=test,
    )


def test_target_overlap_is_not_reported():
    train = pd.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "churn": [0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "customer_id": [104, 105, 106],
            "churn": [1, 0, 1],
        }
    )

    context = make_context(train, test, target="churn")

    findings = SplitOverlapDetector().detect(context)

    assert all(
        "churn" not in finding.affected_columns
        for finding in findings
    )


def test_binary_feature_overlap_is_not_reported():
    train = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105],
            "active": [0, 1, 0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "customer_id": [106, 107, 108, 109, 110],
            "active": [1, 0, 1, 0, 1],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector().detect(context)

    assert all(
        finding.affected_columns != ["active"]
        for finding in findings
    )


def test_low_cardinality_categorical_overlap_is_not_reported():
    train = pd.DataFrame(
        {
            "customer_id": range(101, 121),
            "city": [
                "Delhi",
                "Noida",
                "Delhi",
                "Noida",
                "Delhi",
            ]
            * 4,
        }
    )

    test = pd.DataFrame(
        {
            "customer_id": range(201, 221),
            "city": [
                "Delhi",
                "Noida",
                "Delhi",
                "Noida",
                "Delhi",
            ]
            * 4,
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector().detect(context)

    assert all(
        finding.affected_columns != ["city"]
        for finding in findings
    )


def test_identifier_overlap_is_reported():
    train = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105],
            "value": [10, 20, 30, 40, 50],
        }
    )

    test = pd.DataFrame(
        {
            "customer_id": [105, 106, 107, 108, 109],
            "value": [60, 70, 80, 90, 100],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector(
        columns=["customer_id"]
    ).detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "split_overlap"
    assert findings[0].affected_columns == ["customer_id"]
    assert findings[0].evidence["overlap_count"] == 1

def test_actual_identifier_overlap_is_reported_automatically():
    train = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105],
            "value": [10, 20, 30, 40, 50],
        }
    )

    test = pd.DataFrame(
        {
            "customer_id": [103, 106, 107, 108, 109],
            "value": [60, 70, 80, 90, 100],
        }
    )

    context = make_context(train, test)

    findings = SplitOverlapDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "split_overlap"
    assert findings[0].affected_columns == ["customer_id"]
    assert findings[0].evidence["overlap_count"] == 1
    assert 103 in findings[0].evidence["overlap_values"]