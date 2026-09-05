from __future__ import annotations

from dataleaks.schemas.finding import Finding


class ConfidenceScorer:
    """Utilities for validating and aggregating finding confidence."""

    @staticmethod
    def normalize(confidence: float) -> float:
        """Normalize confidence to the [0.0, 1.0] range."""

        if not isinstance(confidence, (int, float)):
            raise TypeError("confidence must be numeric")

        confidence = float(confidence)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

        return confidence

    @classmethod
    def average(cls, findings: list[Finding]) -> float:
        """Return the average confidence across findings."""

        if not findings:
            return 0.0

        confidences = [
            cls.normalize(finding.confidence)
            for finding in findings
        ]

        return sum(confidences) / len(confidences)

    @classmethod
    def highest(cls, findings: list[Finding]) -> float:
        """Return the highest confidence among findings."""

        if not findings:
            return 0.0

        return max(
            cls.normalize(finding.confidence)
            for finding in findings
        )