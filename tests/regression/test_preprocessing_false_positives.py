from dataleaks.detectors.preprocessing.contamination import (
    PreprocessingContaminationDetector,
)
from dataleaks.detectors.preprocessing.fit_before_split import (
    FitBeforeSplitDetector,
)
from dataleaks.schemas.dataset import DatasetContext
import pandas as pd


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


def test_test_and_validation_contamination_is_critical():
    context = make_context(
        {
            "preprocessing": {
                "contaminated_by": ["test_and_validation"],
            }
        }
    )

    findings = PreprocessingContaminationDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].evidence["contaminated_by"] == [
        "test_and_validation"
    ]


def test_validation_only_contamination_remains_high():
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


def test_train_only_fit_remains_clean():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "train",
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert findings == []


def test_full_dataset_fit_is_critical():
    context = make_context(
        {
            "preprocessing": {
                "fitted_on": "full_dataset",
            }
        }
    )

    findings = FitBeforeSplitDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].confidence == 1.0