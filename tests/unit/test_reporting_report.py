from dataleaks.reporting.report import LeakageReport
from dataleaks.schemas.finding import Finding


def make_finding(
    detector: str = "test_detector",
    severity: str = "high",
) -> Finding:
    return Finding(
        detector=detector,
        category="target_leakage",
        severity=severity,
        confidence=0.9,
        explanation="Test finding",
        recommendation="Fix the issue.",
    )


def test_empty_report_has_no_findings():
    report = LeakageReport()

    assert report.findings == []
    assert report.finding_count == 0
    assert not report.has_leakage


def test_report_stores_findings():
    finding = make_finding()

    report = LeakageReport(findings=[finding])

    assert report.findings == [finding]
    assert report.finding_count == 1
    assert report.has_leakage


def test_report_stores_risk_information():
    report = LeakageReport(
        risk_score=0.85,
        risk_level="critical",
    )

    assert report.risk_score == 0.85
    assert report.risk_level == "critical"


def test_report_stores_confidence_information():
    report = LeakageReport(
        average_confidence=0.75,
        highest_confidence=1.0,
    )

    assert report.average_confidence == 0.75
    assert report.highest_confidence == 1.0


def test_report_stores_recommendations():
    recommendations = [
        "Remove leaking feature.",
        "Split data chronologically.",
    ]

    report = LeakageReport(
        recommendations=recommendations,
    )

    assert report.recommendations == recommendations


def test_report_stores_metadata():
    report = LeakageReport(
        metadata={"rows": 1000},
    )

    assert report.metadata == {"rows": 1000}


def test_findings_by_severity():
    findings = [
        make_finding("detector_a", "critical"),
        make_finding("detector_b", "high"),
        make_finding("detector_c", "critical"),
    ]

    report = LeakageReport(findings=findings)

    critical = report.findings_by_severity("critical")

    assert len(critical) == 2
    assert critical[0].detector == "detector_a"
    assert critical[1].detector == "detector_c"


def test_findings_by_severity_is_case_insensitive():
    report = LeakageReport(
        findings=[make_finding(severity="HIGH")]
    )

    assert len(report.findings_by_severity("high")) == 1


def test_findings_by_severity_strips_whitespace():
    report = LeakageReport(
        findings=[make_finding(severity="high")]
    )

    assert len(report.findings_by_severity("  HIGH  ")) == 1


def test_findings_by_unknown_severity_returns_empty():
    report = LeakageReport(
        findings=[make_finding(severity="high")]
    )

    assert report.findings_by_severity("unknown") == []