from __future__ import annotations

from collections import Counter

from dataleaks.reporting.report import LeakageReport


class ReportSummary:
    """Generate compact summary statistics from a leakage report."""

    @staticmethod
    def build(report: LeakageReport) -> dict[str, object]:
        """Build a structured summary from a LeakageReport."""

        if not isinstance(report, LeakageReport):
            raise TypeError("report must be a LeakageReport")

        severity_counts = Counter(
            finding.severity.strip().lower()
            for finding in report.findings
        )

        return {
            "risk_level": report.risk_level,
            "risk_score": report.risk_score,
            "finding_count": report.finding_count,
            "severity_counts": {
                "critical": severity_counts.get("critical", 0),
                "high": severity_counts.get("high", 0),
                "medium": severity_counts.get("medium", 0),
                "low": severity_counts.get("low", 0),
            },
            "average_confidence": report.average_confidence,
            "highest_confidence": report.highest_confidence,
            "recommendation_count": len(report.recommendations),
            "has_leakage": report.has_leakage,
        }