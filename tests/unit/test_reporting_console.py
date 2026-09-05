import pytest

from dataleaks.reporting.console import ConsoleReporter
from dataleaks.reporting.report import LeakageReport
from dataleaks.schemas.finding import Finding


def make_finding(
    detector: str = "target_direct",
    severity: str = "critical",
) -> Finding:
    return Finding(
        detector=detector,
        category="target_leakage",
        severity=severity,
        confidence=1.0,
        explanation="Feature directly copies the target.",
        recommendation="Remove the leaking feature.",
    )


def test_render_empty_report():
    report = LeakageReport()

    output = ConsoleReporter().render(report)

    assert "DataLeaks Report" in output
    assert "Risk Level: LOW" in output
    assert "Risk Score: 0.00" in output
    assert "Findings: 0" in output


def test_render_risk_information():
    report = LeakageReport(
        risk_score=0.85,
        risk_level="high",
    )

    output = ConsoleReporter().render(report)

    assert "Risk Level: HIGH" in output
    assert "Risk Score: 0.85" in output


def test_render_severity_counts():
    report = LeakageReport(
        findings=[
            make_finding("detector_a", "critical"),
            make_finding("detector_b", "high"),
            make_finding("detector_c", "high"),
            make_finding("detector_d", "medium"),
            make_finding("detector_e", "low"),
        ]
    )

    output = ConsoleReporter().render(report)

    assert "Critical: 1" in output
    assert "High:     2" in output
    assert "Medium:   1" in output
    assert "Low:      1" in output


def test_render_confidence_information():
    report = LeakageReport(
        average_confidence=0.75,
        highest_confidence=1.0,
    )

    output = ConsoleReporter().render(report)

    assert "Average Confidence: 0.75" in output
    assert "Highest Confidence: 1.00" in output


def test_render_recommendation_count():
    report = LeakageReport(
        recommendations=[
            "Fix A.",
            "Fix B.",
        ]
    )

    output = ConsoleReporter().render(report)

    assert "Recommendations: 2" in output


def test_render_findings():
    report = LeakageReport(
        findings=[
            make_finding(),
        ]
    )

    output = ConsoleReporter().render(report)

    assert "Findings:" in output
    assert "[CRITICAL] target_direct" in output
    assert "Feature directly copies the target." in output
    assert "Recommendation: Remove the leaking feature." in output


def test_render_multiple_findings():
    report = LeakageReport(
        findings=[
            make_finding("target_direct", "critical"),
            make_finding("split_duplicates", "high"),
        ]
    )

    output = ConsoleReporter().render(report)

    assert "1. [CRITICAL] target_direct" in output
    assert "2. [HIGH] split_duplicates" in output


def test_render_rejects_invalid_report():
    with pytest.raises(TypeError):
        ConsoleReporter().render("not a report")