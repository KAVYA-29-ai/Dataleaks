from __future__ import annotations

from dataleaks.schemas.finding import Finding
from dataleaks.scoring.severity import Severity


class RiskScorer:
    """Calculate overall leakage risk from findings."""

    SEVERITY_WEIGHTS: dict[str, float] = {
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
        "critical": 1.00,
    }

    @classmethod
    def _finding_score(
        cls,
        finding: Finding,
    ) -> float:
        severity = Severity.normalize(
            finding.severity
        )

        weight = cls.SEVERITY_WEIGHTS[severity]

        confidence = max(
            0.0,
            min(
                1.0,
                float(finding.confidence),
            ),
        )

        return weight * confidence

    @classmethod
    def score(
        cls,
        findings: list[Finding],
    ) -> float:
        """
        Return the strongest weighted finding.

        Overall risk is intentionally determined by the strongest
        individual finding rather than averaging findings together.
        """

        if not findings:
            return 0.0

        return max(
            cls._finding_score(finding)
            for finding in findings
        )

    @classmethod
    def level(
        cls,
        findings: list[Finding],
    ) -> str:
        """Convert the weighted risk score into a risk level."""

        score = cls.score(findings)

        if score >= 1.0:
            return "critical"

        if score >= 0.75:
            return "high"

        if score >= 0.50:
            return "medium"

        return "low"