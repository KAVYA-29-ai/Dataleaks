import pandas as pd
import pytest

from dataleaks.detectors.target.derived import (
    DerivedTargetLeakageDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def test_detects_scaled_target():
    df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40],
            "target_scaled": [20, 40, 60, 80],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["target_scaled"]
    assert findings[0].severity == "critical"


def test_detects_shifted_target():
    df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40],
            "target_shifted": [110, 120, 130, 140],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["target_shifted"]


def test_detects_negative_target_transform():
    df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40],
            "target_inverse": [-10, -20, -30, -40],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["target_inverse"]


def test_does_not_flag_normal_feature():
    df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40],
            "income": [25, 80, 45, 100],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert findings == []


def test_does_not_flag_non_affine_relationship():
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4],
            "squared": [1, 4, 9, 16],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert findings == []


def test_non_numeric_target_is_skipped():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": ["a", "b", "c"],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert findings == []


def test_non_numeric_feature_is_skipped():
    df = pd.DataFrame(
        {
            "target": [1, 2, 3],
            "feature": ["a", "b", "c"],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert findings == []


def test_missing_values_are_handled():
    df = pd.DataFrame(
        {
            "target": [10, 20, 30, 40],
            "derived": [20, 40, None, 80],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DerivedTargetLeakageDetector().detect(context)

    assert len(findings) == 1


def test_no_target_returns_no_findings():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    context = DatasetContext(data=df)

    findings = DerivedTargetLeakageDetector().detect(context)

    assert findings == []


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        DerivedTargetLeakageDetector(tolerance=-1)