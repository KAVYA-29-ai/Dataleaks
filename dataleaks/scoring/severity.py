from __future__ import annotations


class Severity:
    """Utilities for normalizing and comparing finding severity."""

    LEVELS: dict[str, int] = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    @classmethod
    def normalize(cls, severity: str) -> str:
        """Normalize a severity value to lowercase."""
        if not isinstance(severity, str):
            raise TypeError("severity must be a string")

        normalized = severity.strip().lower()

        if normalized not in cls.LEVELS:
            raise ValueError(
                f"Invalid severity '{severity}'. "
                f"Expected one of: {list(cls.LEVELS)}"
            )

        return normalized

    @classmethod
    def score(cls, severity: str) -> int:
        """Return the numeric score for a severity."""
        normalized = cls.normalize(severity)
        return cls.LEVELS[normalized]

    @classmethod
    def is_at_least(
        cls,
        severity: str,
        minimum: str,
    ) -> bool:
        """Return whether severity is at least the minimum level."""
        return cls.score(severity) >= cls.score(minimum)

    @classmethod
    def highest(
        cls,
        *severities: str,
    ) -> str:
        """Return the highest severity among the supplied values."""
        if not severities:
            raise ValueError(
                "At least one severity is required"
            )

        normalized = [
            cls.normalize(severity)
            for severity in severities
        ]

        return max(
            normalized,
            key=cls.LEVELS.__getitem__,
        )