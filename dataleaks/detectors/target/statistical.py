from __future__ import annotations

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class StatisticalTargetLeakageDetector(BaseDetector):
    """Detect features that are suspiciously correlated with the target."""

    name = "target_statistical"

    def __init__(self, threshold: float = 0.999) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        self.threshold = threshold

    def detect(self, context: DatasetContext) -> list[Finding]:
        data = context.data
        target = context.target

        if target is None:
            return []

        if target not in data.columns:
            return []

        target_series = data[target]

        if not pd.api.types.is_numeric_dtype(target_series):
            return []

        findings: list[Finding] = []

        numeric_columns = data.select_dtypes(
            include="number"
        ).columns

        for feature in numeric_columns:
            if feature == target:
                continue

            pair = data[[target, feature]].dropna()

            if len(pair) < 2:
                continue

            target_values = pair[target]
            feature_values = pair[feature]

            # Pearson correlation is undefined when either series
            # has zero variance.
            if target_values.nunique(dropna=True) <= 1:
                continue

            if feature_values.nunique(dropna=True) <= 1:
                continue

            correlation = target_values.corr(feature_values)

            if pd.isna(correlation):
                continue

            strength = abs(float(correlation))

            if strength < self.threshold:
                continue

            findings.append(
                Finding(
                    detector=self.name,
                    category="target_leakage",
                    severity="high",
                    confidence=strength,
                    explanation=(
                        f"Feature '{feature}' is extremely correlated with "
                        f"target '{target}' "
                        f"(|correlation|={strength:.4f}). "
                        "This may indicate target-derived or post-outcome "
                        "information."
                    ),
                    recommendation=(
                        f"Investigate '{feature}' for target leakage and "
                        "remove or transform it if it contains information "
                        "derived from the target."
                    ),
                    affected_columns=[feature],
                    evidence={
                        "feature": feature,
                        "target": target,
                        "absolute_correlation": strength,
                        "correlation": float(correlation),
                        "threshold": self.threshold,
                        "sample_count": len(pair),
                    },
                )
            )

        return findings