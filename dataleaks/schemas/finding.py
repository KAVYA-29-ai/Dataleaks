from dataclasses import dataclass, field
from typing import Any


@dataclass
class Finding:
    """Represents a single potential data-leakage finding."""

    detector: str
    category: str
    severity: str
    confidence: float
    explanation: str
    recommendation: str

    affected_columns: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    related_detectors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.detector.strip():
            raise ValueError("detector must not be empty")

        if not self.category.strip():
            raise ValueError("category must not be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")