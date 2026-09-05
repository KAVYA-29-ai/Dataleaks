import pytest

from dataleaks.schemas.finding import Finding
from dataleaks.scoring.confidence import ConfidenceScorer


def make_finding(confidence: float) -> Finding:
    return Finding(
        detector="test_detector",
        category="test",
        severity="high",
        confidence=confidence,
        explanation="Test finding",
        recommendation="Test recommendation",
    )


def test_normalize_returns_float():
    assert ConfidenceScorer.normalize(0.75) == 0.75


def test_normalize_accepts_integer():
    assert ConfidenceScorer.normalize(1) == 1.0


def test_normalize_accepts_zero():
    assert ConfidenceScorer.normalize(0) == 0.0


def test_normalize_accepts_one():
    assert ConfidenceScorer.normalize(1.0) == 1.0


def test_normalize_rejects_negative_confidence():
    with pytest.raises(ValueError):
        ConfidenceScorer.normalize(-0.1)


def test_normalize_rejects_confidence_above_one():
    with pytest.raises(ValueError):
        ConfidenceScorer.normalize(1.1)


def test_normalize_rejects_non_numeric_confidence():
    with pytest.raises(TypeError):
        ConfidenceScorer.normalize("high")


def test_average_empty_findings():
    assert ConfidenceScorer.average([]) == 0.0


def test_average_single_finding():
    findings = [make_finding(0.8)]

    assert ConfidenceScorer.average(findings) == 0.8


def test_average_multiple_findings():
    findings = [
        make_finding(0.2),
        make_finding(0.6),
        make_finding(1.0),
    ]

    assert ConfidenceScorer.average(findings) == pytest.approx(0.6)


def test_highest_empty_findings():
    assert ConfidenceScorer.highest([]) == 0.0


def test_highest_returns_largest_confidence():
    findings = [
        make_finding(0.2),
        make_finding(0.9),
        make_finding(0.6),
    ]

    assert ConfidenceScorer.highest(findings) == 0.9