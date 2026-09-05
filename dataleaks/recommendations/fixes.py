from __future__ import annotations

from dataleaks.schemas.finding import Finding


class FixEngine:
    """Generate actionable remediation guidance for leakage findings."""

    def suggest(self, finding: Finding) -> str:
        """Return a remediation suggestion for a single finding."""

        if not isinstance(finding, Finding):
            raise TypeError("finding must be a Finding instance")

        if finding.recommendation.strip():
            return finding.recommendation.strip()

        return (
            f"Review the {finding.category} issue reported by "
            f"'{finding.detector}' and remove any information that "
            "would not be legitimately available at prediction time."
        )

    def suggest_many(self, findings: list[Finding]) -> list[str]:
        """Return unique remediation suggestions in finding order."""

        if not isinstance(findings, list):
            raise TypeError("findings must be a list")

        fixes: list[str] = []
        seen: set[str] = set()

        for finding in findings:
            fix = self.suggest(finding)

            if fix in seen:
                continue

            seen.add(fix)
            fixes.append(fix)

        return fixes