from __future__ import annotations

from dataleaks.schemas.finding import Finding


class RecommendationEngine:
    """Generate actionable recommendations from leakage findings."""

    def generate(self, findings: list[Finding]) -> list[str]:
        """Return unique recommendations for the supplied findings."""

        recommendations: list[str] = []
        seen: set[str] = set()

        for finding in findings:
            recommendation = finding.recommendation.strip()

            if not recommendation:
                continue

            if recommendation in seen:
                continue

            seen.add(recommendation)
            recommendations.append(recommendation)

        return recommendations