from __future__ import annotations

import numpy as np
import pandas as pd

from dataleaks import DataLeaks


def run(df: pd.DataFrame, target: str = "target"):
    return DataLeaks(
        df,
        target=target,
    ).run()


def target_findings(report):
    return [
        finding
        for finding in report.findings
        if finding.category == "target_leakage"
    ]


def test_random_numeric_features_are_not_target_leakage():
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "feature_a": rng.normal(size=200),
            "feature_b": rng.normal(size=200),
            "feature_c": rng.normal(size=200),
            "target": rng.integers(0, 2, size=200),
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings == []


def test_perfect_numeric_copy_of_target_is_detected():
    target = np.arange(100)

    df = pd.DataFrame(
        {
            "feature": target.copy(),
            "target": target,
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings

    assert any(
        "feature" in finding.affected_columns
        for finding in findings
    )


def test_scaled_target_is_detected():
    target = np.arange(100)

    df = pd.DataFrame(
        {
            "feature": target * 1000 + 7,
            "target": target,
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings


def test_negative_scaled_target_is_detected():
    target = np.arange(100)

    df = pd.DataFrame(
        {
            "feature": -target,
            "target": target,
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings


def test_constant_feature_is_not_target_leakage():
    df = pd.DataFrame(
        {
            "feature": [5] * 100,
            "target": list(range(100)),
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings == []


def test_single_unique_target_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": np.arange(100),
            "target": [1] * 100,
        }
    )

    report = run(df)

    assert report is not None


def test_small_sample_does_not_create_false_positive():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [1, 2, 3],
        }
    )

    report = run(df)

    assert report is not None


def test_numeric_string_target_does_not_crash():
    df = pd.DataFrame(
        {
            "feature": np.arange(100),
            "target": [str(i % 2) for i in range(100)],
        }
    )

    report = run(df)

    assert report is not None


def test_weak_correlation_is_not_target_leakage():
    rng = np.random.default_rng(123)

    target = rng.normal(size=200)

    feature = (
        0.05 * target
        + rng.normal(size=200)
    )

    df = pd.DataFrame(
        {
            "feature": feature,
            "target": target,
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings == []


def test_near_perfect_but_not_perfect_correlation_is_not_automatically_leakage():
    rng = np.random.default_rng(99)

    target = rng.normal(size=200)

    feature = (
        target
        + rng.normal(
            scale=0.01,
            size=200,
        )
    )

    df = pd.DataFrame(
        {
            "feature": feature,
            "target": target,
        }
    )

    report = run(df)

    findings = target_findings(report)

    assert findings


def test_target_name_variations_do_not_break_detector():
    target = np.arange(100)

    for target_name in [
        "target",
        "label",
        "y",
        "outcome",
        "prediction",
    ]:
        df = pd.DataFrame(
            {
                "feature": target.copy(),
                target_name: target,
            }
        )

        report = run(
            df,
            target=target_name,
        )

        assert target_findings(report)


def test_feature_with_many_missing_values_does_not_crash():
    target = np.arange(100)

    feature = target.astype(float)

    feature[:40] = np.nan

    df = pd.DataFrame(
        {
            "feature": feature,
            "target": target,
        }
    )

    report = run(df)

    assert report is not None


def test_infinite_values_do_not_crash_target_detector():
    target = np.arange(100)

    feature = target.astype(float)

    feature[0] = np.inf
    feature[1] = -np.inf

    df = pd.DataFrame(
        {
            "feature": feature,
            "target": target,
        }
    )

    report = run(df)

    assert report is not None


def test_multiple_features_only_leaking_feature_is_flagged():
    rng = np.random.default_rng(7)

    target = np.arange(100)

    df = pd.DataFrame(
        {
            "random_a": rng.normal(size=100),
            "random_b": rng.normal(size=100),
            "leaky_feature": target.copy(),
            "random_c": rng.normal(size=100),
            "target": target,
        }
    )

    report = run(df)

    findings = target_findings(report)

    affected = {
        column
        for finding in findings
        for column in finding.affected_columns
    }

    assert "leaky_feature" in affected