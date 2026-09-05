from __future__ import annotations

from dataleaks.reporting.report import LeakageReport


class ConsoleReporter:
    """Render a LeakageReport as human-readable console text."""

    def render(self, report: LeakageReport) -> str:
        """Return a formatted console representation of a report."""

        if not isinstance(report, LeakageReport):
            raise TypeError("report must be a LeakageReport")

        schema = report.schema_validation

        summary = [
            "DataLeaks Report",
            "=" * 40,
            f"Risk Level: {report.risk_level.upper()}",
            f"Risk Score: {report.risk_score:.4f}",
            f"Findings: {report.finding_count}",
            "",
            "Severity:",
            f"  Critical: {len(report.findings_by_severity('critical'))}",
            f"  High:     {len(report.findings_by_severity('high'))}",
            f"  Medium:   {len(report.findings_by_severity('medium'))}",
            f"  Low:      {len(report.findings_by_severity('low'))}",
            "",
            f"Average Confidence: {report.average_confidence:.2f}",
            f"Highest Confidence: {report.highest_confidence:.2f}",
            f"Recommendations: {len(report.recommendations)}",
            "",
            "Schema Validation:",
            f"  Mismatch: {'YES' if schema['schema_mismatch'] else 'NO'}",
        ]

        missing_from_train = schema["missing_from_train"]
        missing_from_test = schema["missing_from_test"]
        dtype_mismatches = schema["dtype_mismatches"]
        target = schema["target"]

        if missing_from_train:
            summary.append(
                "  Missing from train: "
                + ", ".join(missing_from_train)
            )

        if missing_from_test:
            summary.append(
                "  Missing from test: "
                + ", ".join(missing_from_test)
            )

        if dtype_mismatches:
            summary.append("  Dtype mismatches:")

            for mismatch in dtype_mismatches:
                summary.append(
                    f"    {mismatch['column']}: "
                    f"train={mismatch['train_dtype']}, "
                    f"test={mismatch['test_dtype']}"
                )

        if not target["train_present"]:
            summary.append(
                "  Target missing from train"
            )

        if not target["test_present"]:
            summary.append(
                "  Target missing from test"
            )

        if report.findings:
            summary.extend([
                "",
                "Findings:",
            ])

            for index, finding in enumerate(
                report.findings,
                start=1,
            ):
                summary.extend([
                    "",
                    f"{index}. [{finding.severity.upper()}] "
                    f"{finding.detector}",
                    f"   {finding.explanation}",
                    f"   Recommendation: {finding.recommendation}",
                ])

        return "\n".join(summary)