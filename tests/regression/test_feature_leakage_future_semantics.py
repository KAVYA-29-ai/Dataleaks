import numpy as np
import pandas as pd

from dataleaks.detectors.feature.suspicious import (
    SuspiciousFeatureDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def test_future_semantic_feature_is_detected():
    target = np.arange(20, dtype=float)

    data = pd.DataFrame(
        {
            "target": target,
            "future_purchase_value": target * 10 + 5,
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
    )

    findings = SuspiciousFeatureDetector().detect(context)

    assert len(findings) == 1

    finding = findings[0]

    assert finding.severity == "high"
    assert finding.evidence["type"] == "future_semantic_feature"
    assert finding.evidence["future_semantic_name_signal"] is True
    assert finding.evidence["absolute_correlation"] >= 0.90


def test_future_semantic_name_alone_does_not_trigger():
    data = pd.DataFrame(
        {
            "target": np.arange(20, dtype=float),
            "future_purchase_value": np.random.default_rng(42).normal(
                size=20
            ),
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
    )

    findings = SuspiciousFeatureDetector().detect(context)

    assert findings == []


def test_generic_future_feature_names_are_supported():
    target = np.arange(20, dtype=float)

    data = pd.DataFrame(
        {
            "target": target,
            "future_revenue": target * 2,
            "next_purchase_value": target * 3,
            "projected_spend": target * 4,
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
    )

    findings = SuspiciousFeatureDetector().detect(context)

    assert {
        finding.evidence["feature"]
        for finding in findings
    } == {
        "future_revenue",
        "next_purchase_value",
        "projected_spend",
    }


def test_future_name_false_positive_is_not_reported():
    data = pd.DataFrame(
        {
            "target": np.arange(20, dtype=float),
            "future_purchase_value": np.random.default_rng(123).normal(
                size=20
            ),
        }
    )

    context = DatasetContext(
        data=data,
        target="target",
    )

    findings = SuspiciousFeatureDetector().detect(context)

    assert not any(
        finding.evidence["feature"]
        == "future_purchase_value"
        for finding in findings
    )