from __future__ import annotations

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class CrossDatasetOverlapDetector(BaseDetector):
    """Detect exact overlap between the current dataset and an external dataset."""

    name = "cross_dataset_overlap"
    category = "cross_dataset_leakage"

    def __init__(
        self,
        reference_data: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        if not isinstance(reference_data, pd.DataFrame):
            raise TypeError(
                "reference_data must be a pandas DataFrame"
            )

        if columns is not None and not columns:
            raise ValueError(
                "columns must contain at least one column"
            )

        self.reference_data = reference_data
        self.columns = columns

    def detect(self, context: DatasetContext) -> list[Finding]:
        data = context.data
        reference = self.reference_data

        if data.empty or reference.empty:
            return []

        columns = self._resolve_columns(
            data,
            reference,
        )

        if not columns:
            return []

        current_rows = (
            data[columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        reference_rows = (
            reference[columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        current_index = pd.MultiIndex.from_frame(current_rows)
        reference_index = pd.MultiIndex.from_frame(reference_rows)

        overlap = current_index.intersection(reference_index)
        overlap_count = len(overlap)

        if overlap_count == 0:
            return []

        current_unique_count = len(current_rows)

        overlap_ratio = (
            overlap_count / current_unique_count
            if current_unique_count
            else 0.0
        )

        if overlap_ratio >= 0.5:
            severity = "critical"
        elif overlap_ratio >= 0.1:
            severity = "high"
        else:
            severity = "medium"

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity=severity,
                confidence=overlap_ratio,
                explanation=(
                    f"{overlap_count} unique row(s) from the current "
                    "dataset also occur in the reference dataset."
                ),
                recommendation=(
                    "Verify whether the reference dataset was used during "
                    "training, preprocessing, augmentation, or evaluation. "
                    "Ensure that shared observations cannot transfer "
                    "target or evaluation information into the model."
                ),
                affected_columns=columns,
                evidence={
                    "type": "cross_dataset_exact_overlap",
                    "overlap_count": overlap_count,
                    "current_unique_rows": current_unique_count,
                    "reference_unique_rows": len(reference_rows),
                    "overlap_ratio": overlap_ratio,
                    "columns": columns,
                },
            )
        ]

    def _resolve_columns(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> list[str]:
        if self.columns is not None:
            missing = [
                column
                for column in self.columns
                if column not in data.columns
                or column not in reference.columns
            ]

            if missing:
                raise ValueError(
                    f"Configured overlap columns are missing: {missing}"
                )

            return self.columns

        return [
            column
            for column in data.columns
            if column in reference.columns
        ]