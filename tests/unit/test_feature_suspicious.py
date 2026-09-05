import pandas as pd
import pytest

from dataleaks.detectors.feature.suspicious import (
    SuspiciousFeatureDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(data, target="target"):
    return DatasetContext(
        data=data,
        target=target,
    )


def test_detects_suspiciously_strong_relationship():
    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "target": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].detector == "feature_suspicious"
    assert findings[0].category == "feature_leakage"
    assert findings[0].severity == "high"
    assert findings[0].confidence == 1.0


def test_strong_but_below_threshold_relationship_is_clean():
    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "target": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                7,
            ],
        }
    )

    findings = SuspiciousFeatureDetector(
        correlation_threshold=0.999
    ).detect(
        make_context(df)
    )

    assert findings == []


def test_negative_perfect_relationship_is_detected():
    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "target": [
                -1,
                -2,
                -3,
                -4,
                -5,
                -6,
                -7,
                -8,
                -9,
                -10,
            ],
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].evidence["correlation"] == -1.0
    assert findings[0].evidence["absolute_correlation"] == 1.0


def test_non_numeric_target_is_ignored():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [
                "a",
                "b",
                "a",
                "b",
                "a",
                "b",
                "a",
                "b",
                "a",
                "b",
            ],
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    assert findings == []


def test_non_numeric_feature_is_ignored():
    df = pd.DataFrame(
        {
            "feature": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
            ],
            "target": range(10),
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    assert findings == []


def test_target_column_is_ignored():
    df = pd.DataFrame(
        {
            "target": range(10),
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    assert findings == []


def test_missing_values_are_ignored():
    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                None,
                None,
                None,
                None,
                None,
            ],
            "target": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    )

    findings = SuspiciousFeatureDetector(
        min_samples=5
    ).detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].evidence["samples"] == 5


def test_insufficient_samples_are_ignored():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [1, 2, 3, 4, 5],
        }
    )

    findings = SuspiciousFeatureDetector(
        min_samples=10
    ).detect(
        make_context(df)
    )

    assert findings == []


def test_constant_feature_is_ignored():
    df = pd.DataFrame(
        {
            "feature": [1] * 10,
            "target": range(10),
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    assert findings == []


def test_missing_target_returns_no_findings():
    df = pd.DataFrame(
        {
            "feature": range(10),
        }
    )

    context = DatasetContext(data=df)

    findings = SuspiciousFeatureDetector().detect(
        context
    )

    assert findings == []


def test_custom_threshold_is_respected():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ],
        }
    )

    findings = SuspiciousFeatureDetector(
        correlation_threshold=1.0
    ).detect(
        make_context(df)
    )

    assert len(findings) == 1


def test_invalid_correlation_threshold_is_rejected():
    with pytest.raises(ValueError):
        SuspiciousFeatureDetector(
            correlation_threshold=1.5
        )


def test_invalid_min_samples_is_rejected():
    with pytest.raises(ValueError):
        SuspiciousFeatureDetector(
            min_samples=1
        )


def test_evidence_contains_required_information():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": range(10),
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(df)
    )

    evidence = findings[0].evidence

    assert evidence["type"] == (
        "suspicious_target_relationship"
    )
    assert evidence["feature"] == "feature"
    assert evidence["target"] == "target"
    assert evidence["samples"] == 10
    assert evidence["threshold"] == 0.999