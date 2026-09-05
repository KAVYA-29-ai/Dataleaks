from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from dataleaks.reporting.report import LeakageReport


class JSONReporter:
    """Serialize a LeakageReport into a stable JSON representation."""

    def to_dict(self, report: LeakageReport) -> dict[str, Any]:
        """Convert a LeakageReport into a JSON-safe dictionary."""

        if not isinstance(report, LeakageReport):
            raise TypeError("report must be a LeakageReport")

        execution_data = []

        for execution in report.execution:
            data = {
                "detector": execution.detector,
                "status": execution.status,
                "finding_count": execution.finding_count,
                "error": execution.error,
                "exception_type": execution.exception_type,
            }

            if execution.reason is not None:
                data["reason"] = execution.reason

            execution_data.append(data)

        return {
            "risk": {
                "score": report.risk_score,
                "level": report.risk_level,
            },
            "summary": {
                "finding_count": report.finding_count,
                "average_confidence": report.average_confidence,
                "highest_confidence": report.highest_confidence,
                "has_leakage": report.has_leakage,
            },
            "severity_counts": {
                "critical": len(
                    report.findings_by_severity("critical")
                ),
                "high": len(
                    report.findings_by_severity("high")
                ),
                "medium": len(
                    report.findings_by_severity("medium")
                ),
                "low": len(
                    report.findings_by_severity("low")
                ),
            },
            "execution": {
                "detectors": execution_data,
                "completed_count": len(
                    report.completed_detectors
                ),
                "failed_count": len(
                    report.failed_detectors
                ),
                "skipped_count": len(
                    report.skipped_detectors
                ),
                "has_warnings": report.has_execution_warnings,
            },
            "schema_validation": {
                "schema_mismatch": report.schema_validation[
                    "schema_mismatch"
                ],
                "missing_from_train": list(
                    report.schema_validation["missing_from_train"]
                ),
                "missing_from_test": list(
                    report.schema_validation["missing_from_test"]
                ),
                "dtype_mismatches": list(
                    report.schema_validation["dtype_mismatches"]
                ),
                "target": dict(
                    report.schema_validation["target"]
                ),
            },
            "findings": [
                asdict(finding)
                for finding in report.findings
            ],
            "recommendations": list(
                report.recommendations
            ),
            "metadata": dict(report.metadata),
        }

    def render(
        self,
        report: LeakageReport,
        *,
        indent: int | None = 2,
    ) -> str:
        """Return the report as a JSON string."""

        data = self.to_dict(report)

        return json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
        )