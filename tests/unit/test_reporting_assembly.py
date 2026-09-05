import pytest

from dataleaks.reporting.report import LeakageReport
from dataleaks.schemas.finding import Finding


def make_finding(
    detector: str,
    severity: str,
    confidence: float,
    recommendation: str,
) -> Finding:
    return Finding(
        detector=detector,
        category="test",
        severity=severity,
        confidence=confidence,
        explanation=f"{detector} explanation",
        recommendation=recommendation,
    )


def test_build_report_from_empty_findings():
    report = LeakageReport.from_findings([])

    assert report.findings == []
    assert report.risk_score == 0.0
    assert report.risk_level == "low"
    assert report.average_confidence == 0.0
    assert report.highest_confidence == 0.0
    assert report.recommendations == []
    assert report.metadata == {}


def test_build_report_calculates_risk():
    findings = [
        make_finding(
            "detector_a",
            "high",
            1.0,
            "Fix A.",
        )
    ]

    report = LeakageReport.from_findings(findings)

    assert report.risk_score == 0.75
    assert report.risk_level == "high"


def test_build_report_calculates_average_confidence():
    findings = [
        make_finding("detector_a", "high", 0.4, "Fix A."),
        make_finding("detector_b", "high", 0.8, "Fix B."),
    ]

    report = LeakageReport.from_findings(findings)

    assert report.average_confidence == pytest.approx(0.6)


def test_build_report_calculates_highest_confidence():
    findings = [
        make_finding("detector_a", "high", 0.4, "Fix A."),
        make_finding("detector_b", "high", 0.9, "Fix B."),
        make_finding("detector_c", "high", 0.7, "Fix C."),
    ]

    report = LeakageReport.from_findings(findings)

    assert report.highest_confidence == 0.9


def test_build_report_generates_recommendations():
    findings = [
        make_finding(
            "detector_a",
            "high",
            0.9,
            "Fix A.",
        ),
        make_finding(
            "detector_b",
            "medium",
            0.8,
            "Fix B.",
        ),
    ]

    report = LeakageReport.from_findings(findings)

    assert report.recommendations == [
        "Fix A.",
        "Fix B.",
    ]


def test_build_report_deduplicates_recommendations():
    findings = [
        make_finding(
            "detector_a",
            "high",
            0.9,
            "Fix this.",
        ),
        make_finding(
            "detector_b",
            "high",
            0.8,
            "Fix this.",
        ),
    ]

    report = LeakageReport.from_findings(findings)

    assert report.recommendations == ["Fix this."]


def test_build_report_preserves_findings():
    findings = [
        make_finding(
            "detector_a",
            "critical",
            1.0,
            "Fix A.",
        )
    ]

    report = LeakageReport.from_findings(findings)

    assert report.findings == findings
    assert report.finding_count == 1
    assert report.has_leakage is True


def test_build_report_preserves_metadata():
    findings = [
        make_finding(
            "detector_a",
            "high",
            0.9,
            "Fix A.",
        )
    ]

    metadata = {
        "rows": 1000,
        "columns": 12,
    }

    report = LeakageReport.from_findings(
        findings,
        metadata=metadata,
    )

    assert report.metadata == metadata


def test_build_report_copies_metadata():
    metadata = {"rows": 1000}

    report = LeakageReport.from_findings(
        [],
        metadata=metadata,
    )

    metadata["rows"] = 2000

    assert report.metadata["rows"] == 1000


def test_build_report_copies_findings_list():
    findings = [
        make_finding(
            "detector_a",
            "high",
            0.9,
            "Fix A.",
        )
    ]

    report = LeakageReport.from_findings(findings)

    findings.clear()

    assert report.finding_count == 1


def test_build_report_rejects_non_list():
    with pytest.raises(TypeError):
        LeakageReport.from_findings(None)


def test_build_report_produces_complete_report():
    findings = [
        make_finding(
            "detector_a",
            "critical",
            1.0,
            "Fix A.",
        ),
        make_finding(
            "detector_b",
            "medium",
            0.6,
            "Fix B.",
        ),
    ]

    report = LeakageReport.from_findings(
        findings,
        metadata={"source": "test"},
    )

    assert report.finding_count == 2
    assert report.has_leakage is True
    assert report.risk_score == 1.0
    assert report.risk_level == "critical"
    assert report.average_confidence == pytest.approx(0.8)
    assert report.highest_confidence == 1.0
    assert report.recommendations == ["Fix A.", "Fix B."]
    assert report.metadata == {"source": "test"}