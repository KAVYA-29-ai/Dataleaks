import pytest

from dataleaks.recommendations.fixes import FixEngine
from dataleaks.schemas.finding import Finding


def make_finding(
    recommendation: str = "Remove the leaking feature.",
) -> Finding:
    return Finding(
        detector="test_detector",
        category="target_leakage",
        severity="high",
        confidence=0.9,
        explanation="Test finding",
        recommendation=recommendation,
    )


def test_suggest_returns_finding_recommendation():
    engine = FixEngine()

    assert engine.suggest(make_finding()) == (
        "Remove the leaking feature."
    )


def test_suggest_strips_whitespace():
    engine = FixEngine()

    finding = make_finding("  Remove the feature.  ")

    assert engine.suggest(finding) == "Remove the feature."


def test_suggest_fallback_when_recommendation_is_blank():
    engine = FixEngine()

    finding = make_finding("   ")

    result = engine.suggest(finding)

    assert "target_leakage" in result
    assert "test_detector" in result


def test_suggest_rejects_invalid_finding():
    engine = FixEngine()

    with pytest.raises(TypeError):
        engine.suggest("not a finding")


def test_suggest_many_empty_list():
    engine = FixEngine()

    assert engine.suggest_many([]) == []


def test_suggest_many_preserves_order():
    engine = FixEngine()

    findings = [
        make_finding("Fix A."),
        make_finding("Fix B."),
        make_finding("Fix C."),
    ]

    assert engine.suggest_many(findings) == [
        "Fix A.",
        "Fix B.",
        "Fix C.",
    ]


def test_suggest_many_removes_duplicates():
    engine = FixEngine()

    findings = [
        make_finding("Fix A."),
        make_finding("Fix B."),
        make_finding("Fix A."),
    ]

    assert engine.suggest_many(findings) == [
        "Fix A.",
        "Fix B.",
    ]


def test_suggest_many_rejects_non_list():
    engine = FixEngine()

    with pytest.raises(TypeError):
        engine.suggest_many(None)