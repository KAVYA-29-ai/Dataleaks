import pandas as pd
import pytest

from dataleaks.schemas.dataset import DatasetContext


def test_matching_train_test_schema_is_valid():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = data.iloc[:2].copy()
    test = data.iloc[2:].copy()

    context = DatasetContext(
        data=data,
        target="target",
        train=train,
        test=test,
    )

    assert context.schema_validation["schema_mismatch"] is False
    assert context.schema_validation["missing_from_train"] == []
    assert context.schema_validation["missing_from_test"] == []
    assert context.schema_validation["dtype_mismatches"] == []
    assert context.schema_validation["target"]["train_present"] is True
    assert context.schema_validation["target"]["test_present"] is True


def test_column_mismatch_is_soft_warning():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [3, 4],
            "other_feature": [10, 20],
            "target": [0, 1],
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
        train=train,
        test=test,
    )

    assert context.schema_validation["schema_mismatch"] is True
    assert context.schema_validation["missing_from_train"] == [
        "other_feature"
    ]
    assert context.schema_validation["missing_from_test"] == []


def test_common_column_dtype_mismatch_is_detected():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature": ["3", "4"],
            "target": [0, 1],
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
        train=train,
        test=test,
    )

    assert context.schema_validation["schema_mismatch"] is True

    dtype_mismatches = context.schema_validation["dtype_mismatches"]

    assert len(dtype_mismatches) == 1

    mismatch = dtype_mismatches[0]

    assert mismatch["column"] == "feature"
    assert mismatch["train_dtype"] == str(
        train["feature"].dtype
    )
    assert mismatch["test_dtype"] == str(
        test["feature"].dtype
    )
    assert mismatch["train_dtype"] != mismatch["test_dtype"]


def test_target_missing_from_test_is_detected_without_hard_failure():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = data.iloc[:2].copy()

    test = pd.DataFrame(
        {
            "feature": [3, 4],
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
        train=train,
        test=test,
    )

    assert context.schema_validation["schema_mismatch"] is True
    assert context.schema_validation["target"]["train_present"] is True
    assert context.schema_validation["target"]["test_present"] is False


def test_no_train_test_schema_validation_defaults_cleanly():
    data = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
    )

    assert context.schema_validation["schema_mismatch"] is False