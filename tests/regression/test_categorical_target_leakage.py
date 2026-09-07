from __future__ import annotations

import numpy as np
import pandas as pd

from dataleaks import DataLeaks


def run(df: pd.DataFrame, target: str = "Churn"):
    return DataLeaks(df, target=target).run()


def target_findings(report):
    return [
        finding
        for finding in report.findings
        if finding.category == "target_leakage"
    ]


def test_categorical_target_encoding_is_detected():
    df = pd.DataFrame(
        {
            "Churn": ["Yes", "No"] * 100,
            "churn_encoded": [1, 0] * 100,
            "noise": np.random.default_rng(42).normal(size=200),
        }
    )

    findings = target_findings(run(df))

    assert any(
        finding.detector == "target_derived"
        and "churn_encoded" in finding.affected_columns
        for finding in findings
    )


def test_categorical_target_copy_is_detected():
    df = pd.DataFrame(
        {
            "Churn": ["Yes", "No"] * 100,
            "churn_copy": ["Yes", "No"] * 100,
        }
    )

    findings = target_findings(run(df))

    assert findings
    assert any("churn_copy" in finding.affected_columns for finding in findings)


def test_categorical_feature_with_strong_target_association_is_detected():
    target = np.array(["Yes"] * 95 + ["No"] * 105)
    feature = np.array(["high"] * 95 + ["low"] * 95 + ["high"] * 10)

    df = pd.DataFrame(
        {
            "Churn": target,
            "risk_bucket": feature,
        }
    )

    findings = target_findings(run(df))

    assert any(
        finding.detector == "target_statistical"
        and "risk_bucket" in finding.affected_columns
        for finding in findings
    )


def test_numeric_feature_with_categorical_target_association_is_detected():
    target = np.array(["Yes"] * 100 + ["No"] * 100)
    feature = np.concatenate(
        [
            np.ones(100) + 0.01,
            np.zeros(100) + 0.01,
        ]
    )

    df = pd.DataFrame(
        {
            "Churn": target,
            "leaky_score": feature,
        }
    )

    findings = target_findings(run(df))

    assert any(
        finding.detector == "target_statistical"
        and "leaky_score" in finding.affected_columns
        for finding in findings
    )


def test_random_categorical_feature_is_not_target_leakage():
    rng = np.random.default_rng(7)

    df = pd.DataFrame(
        {
            "Churn": rng.choice(["Yes", "No"], size=300),
            "region": rng.choice(["north", "south", "east", "west"], size=300),
        }
    )

    findings = target_findings(run(df))

    assert findings == []


def test_non_bijective_categorical_feature_is_not_derived_leakage():
    df = pd.DataFrame(
        {
            "Churn": ["Yes", "No", "Yes", "No"] * 50,
            "bucket": ["risk"] * 100 + ["safe"] * 100,
        }
    )

    findings = target_findings(run(df))

    assert not any(
        finding.detector == "target_derived"
        and "bucket" in finding.affected_columns
        for finding in findings
    )
