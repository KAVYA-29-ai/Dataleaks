import pytest

from dataleaks.schemas.finding import Finding
from dataleaks.scoring.risk import RiskScorer


def make_finding(
    severity: str,
    confidence: float,
) -> Finding:
    return Finding(
        detector="test_detector",
        category="test",
        severity=severity,
        confidence=confidence,
        explanation="Test finding",
        recommendation="Test recommendation",
    )


def test_empty_findings_have_zero_risk():
    assert RiskScorer.score([]) == 0.0


def test_low_risk_finding():
    finding = make_finding("low", 1.0)

    assert RiskScorer.score([finding]) == 0.25


def test_medium_risk_finding():
    finding = make_finding("medium", 1.0)

    assert RiskScorer.score([finding]) == 0.50


def test_high_risk_finding():
    finding = make_finding("high", 1.0)

    assert RiskScorer.score([finding]) == 0.75


def test_critical_risk_finding():
    finding = make_finding("critical", 1.0)

    assert RiskScorer.score([finding]) == 1.0


def test_confidence_reduces_risk():
    finding = make_finding("high", 0.5)

    assert RiskScorer.score([finding]) == 0.375


def test_highest_weighted_finding_determines_risk():
    findings = [
        make_finding("low", 1.0),
        make_finding("critical", 0.25),
        make_finding("medium", 0.5),
    ]

    assert RiskScorer.score(findings) == 0.25


def test_score_is_capped_at_one():
    findings = [
        make_finding("critical", 1.0),
        make_finding("critical", 1.0),
    ]

    assert RiskScorer.score(findings) == 1.0


def test_risk_level_low():
    findings = [
        make_finding("low", 0.5),
    ]

    assert RiskScorer.level(findings) == "low"


def test_risk_level_medium():
    findings = [
        make_finding("medium", 1.0),
    ]

    assert RiskScorer.level(findings) == "medium"


def test_risk_level_high():
    findings = [
        make_finding("high", 1.0),
    ]

    assert RiskScorer.level(findings) == "high"


def test_risk_level_critical():
    findings = [
        make_finding("critical", 1.0),
    ]

    assert RiskScorer.level(findings) == "critical"