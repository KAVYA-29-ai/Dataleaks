import pandas as pd
import pytest

from dataleaks.detectors.preprocessing.fit_before_split import (
    FitBeforeSplitDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(metadata=None):
    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35],
            "income": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        }
    )

    return DatasetContext(
        data=data,
        metadata=metadata or {},
    )


def test_detects_preprocessing_fitted_on_full_dataset():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "full_dataset",
                "transformers": ["StandardScaler"],
                "affected_columns": ["age", "income"],
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "preprocessing_fit_before_split"
    assert findings[0].category == "preprocessing_leakage"
    assert findings[0].severity == "critical"
    assert findings[0].confidence == 1.0


def test_train_only_preprocessing_produces_no_finding():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "train",
                "transformers": ["StandardScaler"],
                "affected_columns": ["age", "income"],
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert findings == []


def test_missing_preprocessing_metadata_produces_no_finding():
    context = make_context()

    findings = FitBeforeSplitDetector().detect(context)

    assert findings == []


def test_missing_fitted_on_produces_no_finding():
    context = make_context(
        {
            "preprocessing": {
                "transformers": ["StandardScaler"],
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert findings == []


def test_case_and_whitespace_are_normalized():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": " FULL_DATASET ",
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert len(findings) == 1


def test_transformers_are_included_in_evidence():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "full_dataset",
                "transformers": [
                    "StandardScaler",
                    "SimpleImputer",
                ],
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert findings[0].evidence["transformers"] == [
        "StandardScaler",
        "SimpleImputer",
    ]


def test_affected_columns_are_preserved():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "full_dataset",
                "affected_columns": ["age", "income"],
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert findings[0].affected_columns == [
        "age",
        "income",
    ]


def test_non_dict_preprocessing_metadata_raises_error():
    context = make_context(
        {
            "preprocessing": "StandardScaler",
        }
    )

    with pytest.raises(TypeError):
        FitBeforeSplitDetector().detect(context)


def test_non_string_fitted_on_raises_error():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": ["train"],
            }
        }
    )

    with pytest.raises(TypeError):
        FitBeforeSplitDetector().detect(context)


def test_non_list_transformers_raises_error():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "full_dataset",
                "transformers": "StandardScaler",
            }
        }
    )

    with pytest.raises(TypeError):
        FitBeforeSplitDetector().detect(context)


def test_non_list_affected_columns_raises_error():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "full_dataset",
                "affected_columns": "age",
            }
        }
    )

    with pytest.raises(TypeError):
        FitBeforeSplitDetector().detect(context)