from __future__ import annotations

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class FitBeforeSplitDetector(BaseDetector):
    """Detect preprocessing fitted on data before train/test splitting."""

    name = "preprocessing_fit_before_split"
    category = "preprocessing_leakage"

    def detect(self, context: DatasetContext) -> list[Finding]:
        preprocessing = context.metadata.get("preprocessing")

        if not preprocessing:
            return []

        if not isinstance(preprocessing, dict):
            raise TypeError(
                "context.metadata['preprocessing'] must be a dictionary"
            )

        fitted_on = preprocessing.get("fitted_on")

        if fitted_on is None:
            return []

        if not isinstance(fitted_on, str):
            raise TypeError(
                "preprocessing 'fitted_on' must be a string"
            )

        normalized = fitted_on.strip().lower()

        if normalized != "full_dataset":
            return []

        transformers = preprocessing.get("transformers", [])

        if transformers is None:
            transformers = []

        if not isinstance(transformers, list):
            raise TypeError(
                "preprocessing 'transformers' must be a list"
            )

        affected_columns = preprocessing.get("affected_columns", [])

        if affected_columns is None:
            affected_columns = []

        if not isinstance(affected_columns, list):
            raise TypeError(
                "preprocessing 'affected_columns' must be a list"
            )

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity="critical",
                confidence=1.0,
                explanation=(
                    "Preprocessing was fitted on the full dataset before "
                    "the train/test split, allowing test-set information "
                    "to influence the learned transformation."
                ),
                recommendation=(
                    "Split the data first, fit preprocessing only on the "
                    "training data, then transform validation and test data "
                    "using the fitted training transformation."
                ),
                affected_columns=affected_columns,
                evidence={
                    "type": "preprocessing_fitted_before_split",
                    "fitted_on": fitted_on,
                    "transformers": transformers,
                },
            )
        ]