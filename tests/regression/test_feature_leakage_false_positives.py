import pandas as pd

from dataleaks.detectors.feature.suspicious import (
    SuspiciousFeatureDetector,
)
from dataleaks.detectors.feature.identifier import (
    IdentifierLeakageDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(data, target=None, metadata=None):
    return DatasetContext(
        data=data,
        target=target,
        metadata=metadata or {},
    )


def test_suspicious_feature_does_not_flag_below_threshold():
    data = pd.DataFrame(
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
                100,
            ],
        }
    )

    findings = SuspiciousFeatureDetector(
        correlation_threshold=0.999,
    ).detect(
        make_context(
            data,
            target="target",
        )
    )

    assert findings == []


def test_suspicious_feature_detects_negative_perfect_relationship():
    data = pd.DataFrame(
        {
            "feature": range(10),
            "target": range(10, 0, -1),
        }
    )

    findings = SuspiciousFeatureDetector().detect(
        make_context(
            data,
            target="target",
        )
    )

    assert len(findings) == 1
    assert findings[0].detector == "feature_suspicious"
    assert findings[0].affected_columns == ["feature"]
    assert findings[0].evidence["absolute_correlation"] == 1.0
    assert findings[0].evidence["correlation"] == -1.0


def test_identifier_requires_identifier_signal():
    data = pd.DataFrame(
        {
            "customer_score": [
                1,
                2,
                3,
                4,
                5,
            ],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(data)
    )

    assert findings == []


def test_explicit_identifier_can_override_name_signal():
    data = pd.DataFrame(
        {
            "customer_key": [
                "a",
                "b",
                "c",
                "d",
                "e",
            ],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(
            data,
            metadata={
                "identifier_columns": ["customer_key"],
            },
        )
    )

    assert len(findings) == 1
    assert findings[0].detector == "feature_identifier"
    assert findings[0].affected_columns == ["customer_key"]
    assert findings[0].evidence["explicit_identifier"] is True
    assert findings[0].evidence["name_matches_identifier_pattern"] is False