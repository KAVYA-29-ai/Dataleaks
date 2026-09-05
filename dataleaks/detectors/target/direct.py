import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class DirectTargetLeakageDetector(BaseDetector):
    """Detect features that directly encode the target."""

    name = "target_direct"
    category = "target_leakage"

    def detect(self, context: DatasetContext) -> list[Finding]:
        if context.target is None:
            return []

        target = context.target
        data = context.data

        findings: list[Finding] = []

        target_values = data[target]

        for column in data.columns:
            if column == target:
                continue

            feature = data[column]

            if self._is_exact_copy(feature, target_values):
                findings.append(
                    Finding(
                        detector=self.name,
                        category=self.category,
                        severity="critical",
                        confidence=1.0,
                        explanation=(
                            f"Feature '{column}' is an exact copy of "
                            f"the target '{target}'."
                        ),
                        recommendation=(
                            f"Remove '{column}' before model training."
                        ),
                        affected_columns=[column],
                        evidence={
                            "type": "exact_copy",
                            "target": target,
                        },
                    )
                )

        return findings

    @staticmethod
    def _is_exact_copy(
        feature: pd.Series,
        target: pd.Series,
    ) -> bool:
        return feature.equals(target)