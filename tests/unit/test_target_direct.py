import pandas as pd

from dataleaks.detectors.target.direct import DirectTargetLeakageDetector
from dataleaks.schemas.dataset import DatasetContext


def test_detects_exact_target_copy():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "target": [0, 1, 0, 1],
            "target_copy": [0, 1, 0, 1],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DirectTargetLeakageDetector().detect(context)

    assert len(findings) == 1
    assert findings[0].affected_columns == ["target_copy"]
    assert findings[0].severity == "critical"
    assert findings[0].confidence == 1.0


def test_does_not_flag_normal_features():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [20000, 50000, 80000, 100000],
            "target": [0, 1, 0, 1],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DirectTargetLeakageDetector().detect(context)

    assert findings == []


def test_does_not_flag_correlated_feature_as_direct_copy():
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "risk_score": [10, 90, 20, 80],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DirectTargetLeakageDetector().detect(context)

    assert findings == []


def test_no_target_returns_no_findings():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "other": [4, 5, 6],
        }
    )

    context = DatasetContext(data=df)

    findings = DirectTargetLeakageDetector().detect(context)

    assert findings == []


def test_multiple_target_copies_are_detected():
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "copy_a": [0, 1, 0, 1],
            "copy_b": [0, 1, 0, 1],
            "normal": [10, 20, 30, 40],
        }
    )

    context = DatasetContext(data=df, target="target")

    findings = DirectTargetLeakageDetector().detect(context)

    assert len(findings) == 2
    assert {f.affected_columns[0] for f in findings} == {
        "copy_a",
        "copy_b",
    }