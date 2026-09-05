from __future__ import annotations

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class PreprocessingContaminationDetector(BaseDetector):
    """Detect contamination of training preprocessing with held-out data."""

    name = "preprocessing_contamination"
    category = "preprocessing_leakage"

    def detect(self, context: DatasetContext) -> list[Finding]:
        preprocessing = context.metadata.get("preprocessing")

        if not preprocessing:
            return []

        if not isinstance(preprocessing, dict):
            raise TypeError(
                "context.metadata['preprocessing'] must be a dictionary"
            )

        contaminated_by = preprocessing.get("contaminated_by")

        if contaminated_by is None:
            return []

        if not isinstance(contaminated_by, list):
            raise TypeError(
                "preprocessing 'contaminated_by' must be a list"
            )

        contaminated_by = [
            str(source).strip().lower()
            for source in contaminated_by
            if str(source).strip()
        ]

        held_out_sources = {
            "test",
            "validation",
            "test_and_validation",
        }

        detected_sources = [
            source
            for source in contaminated_by
            if source in held_out_sources
        ]

        if not detected_sources:
            return []

        affected_columns = preprocessing.get("affected_columns", [])

        if affected_columns is None:
            affected_columns = []

        if not isinstance(affected_columns, list):
            raise TypeError(
                "preprocessing 'affected_columns' must be a list"
            )

        severity = (
            "critical"
            if any(source in {"test", "test_and_validation"} for source in detected_sources)
            else "high"
        )

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity=severity,
                confidence=1.0,
                explanation=(
                    "Training preprocessing was contaminated by held-out "
                    f"data source(s): {', '.join(detected_sources)}."
                ),
                recommendation=(
                    "Fit preprocessing using training data only. Do not "
                    "use validation or test observations when learning "
                    "imputation, scaling, encoding, normalization, or "
                    "other transformation parameters."
                ),
                affected_columns=affected_columns,
                evidence={
                    "type": "held_out_data_contamination",
                    "contaminated_by": detected_sources,
                },
            )
        ]