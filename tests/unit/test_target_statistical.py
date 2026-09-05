import pandas as pd
import pytest

from dataleaks.detectors.target.statistical import (
    StatisticalTargetLeakageDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def test_detects_perfect_positive_association():
    df = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40, 50],
            "target": [1, 2, 3, 4, 5],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["feature"]
    assert findings[0].severity == "high"
    assert findings[0].confidence == 1.0


def test_detects_perfect_negative_association():
    df = pd.DataFrame(
        {
            "feature": [50, 40, 30, 20, 10],
            "target": [1, 2, 3, 4, 5],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["feature"]


def test_does_not_flag_normal_association():
    df = pd.DataFrame(
        {
            "feature": [1, 4, 2, 8, 3, 7],
            "target": [2, 1, 5, 3, 8, 4],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert findings == []


def test_non_numeric_target_is_skipped():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": ["yes", "no", "yes", "no"],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert findings == []


def test_non_numeric_feature_is_skipped():
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "target": [1, 2, 3, 4],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert findings == []


def test_missing_values_are_handled():
    df = pd.DataFrame(
        {
            "feature": [10, 20, None, 40, 50],
            "target": [1, 2, 3, 4, 5],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert len(findings) == 1


def test_custom_threshold():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [1, 2, 3, 5, 8],
        }
    )

    context = DatasetContext(data=df, target="target")

    detector = StatisticalTargetLeakageDetector(threshold=0.99)

    findings = detector.detect(context)

    assert isinstance(findings, list)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        StatisticalTargetLeakageDetector(threshold=1.5)


def test_no_target_returns_no_findings():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    context = DatasetContext(data=df)

    findings = StatisticalTargetLeakageDetector().detect(context)

    assert findings == []