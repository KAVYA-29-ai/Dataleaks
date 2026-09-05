import pytest

from dataleaks.engine.aggregator import FindingAggregator
from dataleaks.schemas.finding import Finding
from dataleaks.scoring.confidence import ConfidenceScorer
from dataleaks.scoring.risk import RiskScorer


def make_finding(
    detector: str = "test_detector",
    *,
    severity: str = "high",
    confidence: float = 0.9,
    feature: str = "feature_a",
    target: str = "target",
    correlation: float | None = None,
) -> Finding:
    evidence = {
        "feature": feature,
        "target": target,
    }

    if correlation is not None:
        evidence["absolute_correlation"] = correlation

    return Finding(
        detector=detector,
        category="test",
        severity=severity,
        confidence=confidence,
        explanation=f"{detector} explanation",
        recommendation=f"Fix {feature}.",
        affected_columns=[feature],
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------


def test_risk_score_zero_confidence_is_zero():
    finding = make_finding(
        severity="critical",
        confidence=0.0,
    )

    assert RiskScorer.score([finding]) == 0.0
    assert RiskScorer.level([finding]) == "low"


def test_risk_score_uses_strongest_weighted_finding():
    findings = [
        make_finding(
            severity="low",
            confidence=1.0,
        ),
        make_finding(
            severity="medium",
            confidence=0.4,
        ),
        make_finding(
            severity="high",
            confidence=0.8,
        ),
    ]

    assert RiskScorer.score(findings) == pytest.approx(0.6)


def test_risk_score_is_independent_of_finding_order():
    findings = [
        make_finding(
            detector="a",
            severity="high",
            confidence=0.8,
        ),
        make_finding(
            detector="b",
            severity="critical",
            confidence=0.4,
        ),
        make_finding(
            detector="c",
            severity="medium",
            confidence=1.0,
        ),
    ]

    forward = RiskScorer.score(findings)
    reverse = RiskScorer.score(list(reversed(findings)))

    assert forward == reverse
    assert forward == pytest.approx(0.6)


def test_risk_level_boundary_at_medium():
    finding = make_finding(
        severity="medium",
        confidence=1.0,
    )

    assert RiskScorer.score([finding]) == 0.5
    assert RiskScorer.level([finding]) == "medium"


def test_risk_level_boundary_at_high():
    finding = make_finding(
        severity="high",
        confidence=1.0,
    )

    assert RiskScorer.score([finding]) == 0.75
    assert RiskScorer.level([finding]) == "high"


def test_risk_level_boundary_at_critical():
    finding = make_finding(
        severity="critical",
        confidence=1.0,
    )

    assert RiskScorer.score([finding]) == 1.0
    assert RiskScorer.level([finding]) == "critical"


def test_critical_finding_with_partial_confidence_does_not_become_critical():
    finding = make_finding(
        severity="critical",
        confidence=0.99,
    )

    assert RiskScorer.score([finding]) == pytest.approx(0.99)
    assert RiskScorer.level([finding]) == "high"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def test_confidence_average_is_order_independent():
    findings = [
        make_finding(confidence=0.2),
        make_finding(confidence=0.8),
        make_finding(confidence=1.0),
    ]

    forward = ConfidenceScorer.average(findings)
    reverse = ConfidenceScorer.average(list(reversed(findings)))

    assert forward == pytest.approx(0.6666666667)
    assert forward == reverse


def test_highest_confidence_is_order_independent():
    findings = [
        make_finding(confidence=0.2),
        make_finding(confidence=0.95),
        make_finding(confidence=0.6),
    ]

    assert ConfidenceScorer.highest(findings) == 0.95
    assert ConfidenceScorer.highest(
        list(reversed(findings))
    ) == 0.95


def test_confidence_normalization_preserves_boundary_values():
    assert ConfidenceScorer.normalize(0.0) == 0.0
    assert ConfidenceScorer.normalize(1.0) == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_compatible_findings_merge_regardless_of_input_order():
    first = make_finding(
        detector="target_statistical",
        confidence=0.9,
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        confidence=0.8,
        correlation=0.9998,
    )

    aggregator = FindingAggregator()

    forward = aggregator.aggregate([first, second])
    reverse = aggregator.aggregate([second, first])

    assert len(forward) == 1
    assert len(reverse) == 1

    assert set(forward[0].related_detectors) == {
        "target_statistical",
        "feature_suspicious",
    }

    assert set(reverse[0].related_detectors) == {
        "target_statistical",
        "feature_suspicious",
    }


def test_aggregation_keeps_strongest_severity():
    first = make_finding(
        detector="target_statistical",
        severity="medium",
        confidence=0.7,
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        severity="critical",
        confidence=0.8,
        correlation=0.9998,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert len(result) == 1
    assert result[0].severity == "critical"


def test_aggregation_keeps_strongest_confidence():
    first = make_finding(
        detector="target_statistical",
        confidence=0.7,
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        confidence=0.95,
        correlation=0.9998,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert len(result) == 1
    assert result[0].confidence == 0.95


def test_aggregation_unions_affected_columns():
    first = make_finding(
        detector="target_statistical",
        feature="feature_a",
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        feature="feature_a",
        correlation=0.9998,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert result[0].affected_columns == ["feature_a"]


def test_unrelated_findings_never_merge():
    first = make_finding(
        detector="target_statistical",
        feature="feature_a",
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        feature="feature_b",
        correlation=0.9998,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert len(result) == 2


def test_findings_with_different_targets_never_merge():
    first = make_finding(
        detector="target_statistical",
        target="churn",
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        target="revenue",
        correlation=0.9998,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert len(result) == 2


def test_findings_with_different_correlations_do_not_merge():
    first = make_finding(
        detector="target_statistical",
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        correlation=0.95,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert len(result) == 2


def test_non_mergeable_findings_preserve_input_order():
    first = make_finding(
        detector="detector_a",
        feature="feature_a",
    )

    second = make_finding(
        detector="detector_b",
        feature="feature_b",
    )

    third = make_finding(
        detector="detector_c",
        feature="feature_c",
    )

    result = FindingAggregator().aggregate(
        [first, second, third]
    )

    assert [finding.detector for finding in result] == [
        "detector_a",
        "detector_b",
        "detector_c",
    ]


def test_aggregated_finding_preserves_evidence_sources():
    first = make_finding(
        detector="target_statistical",
        correlation=0.9998,
    )

    second = make_finding(
        detector="feature_suspicious",
        correlation=0.9998,
    )

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert len(result) == 1
    assert result[0].evidence["type"] == "aggregated_findings"
    assert len(result[0].evidence["sources"]) == 2


def test_aggregated_finding_preserves_metadata():
    first = make_finding(
        detector="target_statistical",
        correlation=0.9998,
    )
    first.metadata = {"source_a": "test"}

    second = make_finding(
        detector="feature_suspicious",
        correlation=0.9998,
    )
    second.metadata = {"source_b": "test"}

    result = FindingAggregator().aggregate(
        [first, second]
    )

    assert result[0].metadata == {
        "source_a": "test",
        "source_b": "test",
    }