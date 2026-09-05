from __future__ import annotations

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class TargetEncodingLeakageDetector(BaseDetector):
    """Detect target encoding that was computed using held-out data."""

    name = "feature_target_encoding"
    category = "feature_leakage"

    def detect(self, context: DatasetContext) -> list[Finding]:
        encoding = context.metadata.get("target_encoding")

        if not encoding:
            return []

        if not isinstance(encoding, dict):
            raise TypeError(
                "context.metadata['target_encoding'] must be a dictionary"
            )

        fitted_on = encoding.get("fitted_on")

        if fitted_on is None:
            return []

        if not isinstance(fitted_on, str):
            raise TypeError(
                "target_encoding 'fitted_on' must be a string"
            )

        normalized = fitted_on.strip().lower()

        if normalized not in {
            "full_dataset",
            "test",
            "validation",
            "test_and_validation",
        }:
            return []

        columns = encoding.get("columns", [])

        if columns is None:
            columns = []

        if not isinstance(columns, list):
            raise TypeError(
                "target_encoding 'columns' must be a list"
            )

        if normalized in {
            "test",
            "validation",
            "test_and_validation",
        }:
            severity = "critical"
        else:
            severity = "high"

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity=severity,
                confidence=1.0,
                explanation=(
                    "Target encoding was fitted using data outside the "
                    "training split, allowing target information from "
                    "held-out observations to influence the encoded "
                    "features."
                ),
                recommendation=(
                    "Fit target encoding using training data only. "
                    "Apply the learned encoding to validation and test "
                    "data without using their target values."
                ),
                affected_columns=columns,
                evidence={
                    "type": "target_encoding_held_out_contamination",
                    "fitted_on": normalized,
                    "columns": columns,
                },
            )
        ]