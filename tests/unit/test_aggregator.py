from dataleaks.engine.aggregator import FindingAggregator
from dataleaks.schemas.finding import Finding


def make_finding(
    detector: str,
    *,
    feature: str = "target_leak_direct",
    target: str = "churn",
    correlation: float = 0.9998,
) -> Finding:
    return Finding(
        detector=detector,
        category="target_leakage",
        severity="high",
        confidence=correlation,
        explanation=f"{detector} detected {feature}.",
        recommendation=f"Review {feature}.",
        affected_columns=[feature],
        evidence={
            "feature": feature,
            "target": target,
            "absolute_correlation": correlation,
        },
    )


def test_non_mergeable_findings_are_preserved():
    findings = [
        make_finding("target_statistical"),
        make_finding(
            "feature_identifier",
            feature="customer_id",
        ),
    ]

    result = FindingAggregator().aggregate(findings)

    assert len(result) == 2
    assert result[0].detector == "target_statistical"
    assert result[1].detector == "feature_identifier"


def test_mergeable_pair_is_combined():
    findings = [
        make_finding("target_statistical"),
        make_finding("feature_suspicious"),
    ]

    result = FindingAggregator().aggregate(findings)

    assert len(result) == 1


def test_merged_finding_preserves_related_detectors():
    findings = [
        make_finding("target_statistical"),
        make_finding("feature_suspicious"),
    ]

    result = FindingAggregator().aggregate(findings)

    assert len(result) == 1

    merged = result[0]

    assert set(merged.related_detectors) == {
        "target_statistical",
        "feature_suspicious",
    }


def test_different_features_are_not_merged():
    findings = [
        make_finding(
            "target_statistical",
            feature="feature_a",
        ),
        make_finding(
            "feature_suspicious",
            feature="feature_b",
        ),
    ]

    result = FindingAggregator().aggregate(findings)

    assert len(result) == 2


def test_different_targets_are_not_merged():
    findings = [
        make_finding(
            "target_statistical",
            target="churn",
        ),
        make_finding(
            "feature_suspicious",
            target="revenue",
        ),
    ]

    result = FindingAggregator().aggregate(findings)

    assert len(result) == 2


def test_three_way_merge_is_not_supported_in_v1():
    findings = [
        make_finding("target_statistical"),
        make_finding("feature_suspicious"),
        make_finding("target_derived"),
    ]

    result = FindingAggregator().aggregate(findings)

    # V1 intentionally supports configured pairwise aggregation,
    # not arbitrary graph/transitive clustering.
    assert len(result) >= 2