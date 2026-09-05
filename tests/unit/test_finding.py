import pytest

from dataleaks.schemas.finding import Finding


def test_finding_creation():
    finding = Finding(
        detector="target_direct",
        category="target_leakage",
        severity="critical",
        confidence=0.95,
        explanation="Feature directly encodes the target.",
        recommendation="Remove the feature before model training.",
        affected_columns=["target_proxy"],
        evidence={"association": 0.95},
    )

    assert finding.detector == "target_direct"
    assert finding.category == "target_leakage"
    assert finding.severity == "critical"
    assert finding.confidence == 0.95
    assert finding.affected_columns == ["target_proxy"]


def test_confidence_must_be_valid():
    with pytest.raises(ValueError):
        Finding(
            detector="test",
            category="test",
            severity="low",
            confidence=1.5,
            explanation="test",
            recommendation="test",
        )


def test_detector_must_not_be_empty():
    with pytest.raises(ValueError):
        Finding(
            detector="",
            category="test",
            severity="low",
            confidence=0.5,
            explanation="test",
            recommendation="test",
        )