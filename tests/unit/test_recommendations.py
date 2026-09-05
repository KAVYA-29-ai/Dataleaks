from dataleaks.recommendations.recommendations import RecommendationEngine
from dataleaks.schemas.finding import Finding


def make_finding(recommendation: str) -> Finding:
    return Finding(
        detector="test_detector",
        category="test",
        severity="high",
        confidence=0.9,
        explanation="Test finding",
        recommendation=recommendation,
    )


def test_empty_findings_return_empty_recommendations():
    engine = RecommendationEngine()

    assert engine.generate([]) == []


def test_single_recommendation_is_returned():
    engine = RecommendationEngine()

    findings = [
        make_finding("Remove the leaking feature."),
    ]

    assert engine.generate(findings) == [
        "Remove the leaking feature."
    ]


def test_multiple_recommendations_are_returned():
    engine = RecommendationEngine()

    findings = [
        make_finding("Remove feature A."),
        make_finding("Split data chronologically."),
    ]

    assert engine.generate(findings) == [
        "Remove feature A.",
        "Split data chronologically.",
    ]


def test_duplicate_recommendations_are_removed():
    engine = RecommendationEngine()

    findings = [
        make_finding("Remove the leaking feature."),
        make_finding("Remove the leaking feature."),
    ]

    assert engine.generate(findings) == [
        "Remove the leaking feature."
    ]


def test_duplicate_detection_preserves_order():
    engine = RecommendationEngine()

    findings = [
        make_finding("Fix A."),
        make_finding("Fix B."),
        make_finding("Fix A."),
        make_finding("Fix C."),
        make_finding("Fix B."),
    ]

    assert engine.generate(findings) == [
        "Fix A.",
        "Fix B.",
        "Fix C.",
    ]


def test_blank_recommendations_are_ignored():
    engine = RecommendationEngine()

    findings = [
        make_finding("   "),
        make_finding("Fix leakage."),
    ]

    assert engine.generate(findings) == [
        "Fix leakage."
    ]


def test_whitespace_is_stripped():
    engine = RecommendationEngine()

    findings = [
        make_finding("  Fix leakage.  "),
    ]

    assert engine.generate(findings) == [
        "Fix leakage."
    ]