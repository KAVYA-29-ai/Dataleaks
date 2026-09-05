import pandas as pd
import pytest

from dataleaks.detectors.feature.target_encoding import (
    TargetEncodingLeakageDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(metadata=None):
    data = pd.DataFrame(
        {
            "category": ["A", "B", "A", "C"],
            "target": [1, 0, 1, 0],
        }
    )

    return DatasetContext(
        data=data,
        target="target",
        metadata=metadata or {},
    )


def test_detects_full_dataset_target_encoding():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "full_dataset",
                "columns": ["category"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "feature_target_encoding"
    assert findings[0].category == "feature_leakage"
    assert findings[0].severity == "high"
    assert findings[0].confidence == 1.0


def test_detects_test_target_encoding():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "test",
                "columns": ["category"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_detects_validation_target_encoding():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "validation",
                "columns": ["category"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_train_only_target_encoding_is_clean():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "train",
                "columns": ["category"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert findings == []


def test_missing_metadata_is_clean():
    context = make_context()

    findings = TargetEncodingLeakageDetector().detect(context)

    assert findings == []


def test_missing_fitted_on_is_clean():
    context = make_context(
        {
            "target_encoding": {
                "columns": ["category"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert findings == []


def test_case_and_whitespace_are_normalized():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": " TEST ",
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_columns_are_preserved():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "full_dataset",
                "columns": ["category", "region"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert findings[0].affected_columns == [
        "category",
        "region",
    ]


def test_evidence_contains_fitted_on():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "full_dataset",
                "columns": ["category"],
            }
        }
    )

    findings = TargetEncodingLeakageDetector().detect(context)

    assert (
        findings[0].evidence["fitted_on"]
        == "full_dataset"
    )


def test_non_dict_metadata_raises_error():
    context = make_context(
        {
            "target_encoding": "category",
        }
    )

    with pytest.raises(TypeError):
        TargetEncodingLeakageDetector().detect(context)


def test_non_string_fitted_on_raises_error():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": ["train"],
            }
        }
    )

    with pytest.raises(TypeError):
        TargetEncodingLeakageDetector().detect(context)


def test_non_list_columns_raises_error():
    context = make_context(
        {
            "target_encoding": {
                "fitted_on": "full_dataset",
                "columns": "category",
            }
        }
    )

    with pytest.raises(TypeError):
        TargetEncodingLeakageDetector().detect(context)