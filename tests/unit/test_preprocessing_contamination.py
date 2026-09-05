import pandas as pd
import pytest

from dataleaks.detectors.preprocessing.contamination import (
    PreprocessingContaminationDetector,
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


def test_detects_test_data_contamination():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": ["test"],
                "affected_columns": ["age", "income"],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "preprocessing_contamination"
    assert findings[0].category == "preprocessing_leakage"
    assert findings[0].severity == "critical"
    assert findings[0].confidence == 1.0


def test_detects_validation_contamination():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": ["validation"],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "high"


def test_test_and_validation_contamination_is_detected():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": [
                    "test",
                    "validation",
                ],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_train_only_preprocessing_is_clean():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": ["train"],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert findings == []


def test_missing_metadata_produces_no_finding():
    context = make_context()

    findings = PreprocessingContaminationDetector().detect(context)

    assert findings == []


def test_missing_contamination_field_produces_no_finding():
    context = make_context(
        {
            "preprocessing": {
                "affected_columns": ["age"],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert findings == []


def test_empty_contamination_list_is_clean():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": [],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert findings == []


def test_case_and_whitespace_are_normalized():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": [
                    " TEST ",
                ],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert len(findings) == 1


def test_irrelevant_sources_are_ignored():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": [
                    "train",
                    "external_reference",
                ],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert findings == []


def test_non_list_contaminated_by_raises_error():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": "test",
            }
        }
    )

    with pytest.raises(TypeError):
        PreprocessingContaminationDetector().detect(context)


def test_non_list_affected_columns_raises_error():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": ["test"],
                "affected_columns": "age",
            }
        }
    )

    with pytest.raises(TypeError):
        PreprocessingContaminationDetector().detect(context)