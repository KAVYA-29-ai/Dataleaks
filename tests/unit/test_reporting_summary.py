import pytest

from dataleaks.reporting.report import LeakageReport
from dataleaks.reporting.summary import ReportSummary
from dataleaks.schemas.finding import Finding


def make_finding(severity: str) -> Finding:
    return Finding(
        detector="test_detector",
        category="test",
        severity=severity,
        confidence=0.9,
        explanation="Test finding",
        recommendation="Fix the issue.",
    )


def test_empty_report_summary():
    report = LeakageReport()

    summary = ReportSummary.build(report)

    assert summary == {
        "risk_level": "low",
        "risk_score": 0.0,
        "finding_count": 0,
        "severity_counts": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        },
        "average_confidence": 0.0,
        "highest_confidence": 0.0,
        "recommendation_count": 0,
        "has_leakage": False,
    }


def test_summary_counts_findings():
    report = LeakageReport(
        findings=[
            make_finding("critical"),
            make_finding("high"),
            make_finding("high"),
            make_finding("medium"),
            make_finding("low"),
        ]
    )

    summary = ReportSummary.build(report)

    assert summary["finding_count"] == 5
    assert summary["severity_counts"] == {
        "critical": 1,
        "high": 2,
        "medium": 1,
        "low": 1,
    }


def test_summary_preserves_risk_information():
    report = LeakageReport(
        risk_score=0.82,
        risk_level="high",
    )

    summary = ReportSummary.build(report)

    assert summary["risk_score"] == 0.82
    assert summary["risk_level"] == "high"


def test_summary_preserves_confidence_information():
    report = LeakageReport(
        average_confidence=0.72,
        highest_confidence=1.0,
    )

    summary = ReportSummary.build(report)

    assert summary["average_confidence"] == 0.72
    assert summary["highest_confidence"] == 1.0


def test_summary_counts_recommendations():
    report = LeakageReport(
        recommendations=[
            "Fix A.",
            "Fix B.",
            "Fix C.",
        ]
    )

    summary = ReportSummary.build(report)

    assert summary["recommendation_count"] == 3


def test_summary_reports_leakage_presence():
    report = LeakageReport(
        findings=[make_finding("high")]
    )

    summary = ReportSummary.build(report)

    assert summary["has_leakage"] is True


def test_summary_is_case_insensitive_for_severity():
    report = LeakageReport(
        findings=[
            make_finding("HIGH"),
            make_finding("Critical"),
        ]
    )

    summary = ReportSummary.build(report)

    assert summary["severity_counts"]["high"] == 1
    assert summary["severity_counts"]["critical"] == 1


def test_summary_rejects_invalid_report():
    with pytest.raises(TypeError):
        ReportSummary.build("not a report")