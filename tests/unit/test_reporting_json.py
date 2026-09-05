import json

import pytest

from dataleaks.reporting.json import JSONReporter
from dataleaks.reporting.report import LeakageReport
from dataleaks.schemas.finding import Finding


def make_finding() -> Finding:
    return Finding(
        detector="target_direct",
        category="target_leakage",
        severity="critical",
        confidence=1.0,
        explanation="Feature copies the target.",
        recommendation="Remove the feature.",
        affected_columns=["leaked_feature"],
        evidence={
            "type": "exact_copy",
            "target": "target",
        },
        metadata={
            "source": "test",
        },
    )


def test_to_dict_empty_report():
    report = LeakageReport()

    data = JSONReporter().to_dict(report)

    assert data["risk"] == {
        "score": 0.0,
        "level": "low",
    }

    assert data["summary"]["finding_count"] == 0
    assert data["summary"]["has_leakage"] is False
    assert data["findings"] == []
    assert data["recommendations"] == []


def test_to_dict_contains_finding():
    report = LeakageReport(
        findings=[make_finding()],
    )

    data = JSONReporter().to_dict(report)

    assert len(data["findings"]) == 1

    finding = data["findings"][0]

    assert finding["detector"] == "target_direct"
    assert finding["category"] == "target_leakage"
    assert finding["severity"] == "critical"
    assert finding["confidence"] == 1.0
    assert finding["affected_columns"] == ["leaked_feature"]


def test_to_dict_contains_evidence_and_metadata():
    report = LeakageReport(
        findings=[make_finding()],
    )

    data = JSONReporter().to_dict(report)

    finding = data["findings"][0]

    assert finding["evidence"]["type"] == "exact_copy"
    assert finding["metadata"]["source"] == "test"


def test_to_dict_contains_severity_counts():
    report = LeakageReport(
        findings=[
            make_finding(),
            Finding(
                detector="detector_b",
                category="split_leakage",
                severity="high",
                confidence=0.8,
                explanation="Overlap detected.",
                recommendation="Separate the datasets.",
            ),
        ],
    )

    data = JSONReporter().to_dict(report)

    assert data["severity_counts"] == {
        "critical": 1,
        "high": 1,
        "medium": 0,
        "low": 0,
    }


def test_to_dict_contains_recommendations():
    report = LeakageReport(
        recommendations=[
            "Remove the feature.",
            "Split the data.",
        ],
    )

    data = JSONReporter().to_dict(report)

    assert data["recommendations"] == [
        "Remove the feature.",
        "Split the data.",
    ]


def test_render_returns_valid_json():
    report = LeakageReport(
        findings=[make_finding()],
        risk_score=1.0,
        risk_level="critical",
    )

    output = JSONReporter().render(report)

    parsed = json.loads(output)

    assert parsed["risk"]["score"] == 1.0
    assert parsed["risk"]["level"] == "critical"
    assert len(parsed["findings"]) == 1


def test_render_supports_compact_json():
    report = LeakageReport()

    output = JSONReporter().render(
        report,
        indent=None,
    )

    parsed = json.loads(output)

    assert parsed["findings"] == []
    assert "\n" not in output


def test_render_preserves_unicode():
    report = LeakageReport(
        metadata={"message": "डेटा leakage"},
    )

    output = JSONReporter().render(report)

    assert "डेटा leakage" in output


def test_to_dict_rejects_invalid_report():
    with pytest.raises(TypeError):
        JSONReporter().to_dict("not a report")


def test_render_rejects_invalid_report():
    with pytest.raises(TypeError):
        JSONReporter().render("not a report")