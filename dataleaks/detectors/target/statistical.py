from __future__ import annotations

import math

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class StatisticalTargetLeakageDetector(BaseDetector):
    """Detect features that are suspiciously associated with the target."""

    name = "target_statistical"

    def __init__(
        self,
        threshold: float = 0.999,
        categorical_threshold: float = 0.75,
        max_categories: int = 50,
        min_samples: int = 20,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if not 0.0 <= categorical_threshold <= 1.0:
            raise ValueError(
                "categorical_threshold must be between 0.0 and 1.0"
            )
        if max_categories < 2:
            raise ValueError("max_categories must be at least 2")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")

        self.threshold = threshold
        self.categorical_threshold = categorical_threshold
        self.max_categories = max_categories
        self.min_samples = min_samples

    def detect(self, context: DatasetContext) -> list[Finding]:
        data = context.data
        target = context.target

        if target is None or target not in data.columns:
            return []

        target_series = data[target]
        findings: list[Finding] = []

        for feature in data.columns:
            if feature == target:
                continue

            pair = data[[target, feature]].dropna()

            if len(pair) < self.min_samples:
                continue

            target_values = pair[target]
            feature_values = pair[feature]

            if target_values.nunique(dropna=True) <= 1:
                continue
            if feature_values.nunique(dropna=True) <= 1:
                continue

            target_numeric = self._is_numeric(target_values)
            feature_numeric = self._is_numeric(feature_values)

            if target_numeric and feature_numeric:
                strength = self._pearson_strength(
                    target_values,
                    feature_values,
                )
                threshold = self.threshold
                metric = "absolute_correlation"
            elif target_numeric or feature_numeric:
                categorical_values = (
                    feature_values if target_numeric else target_values
                )
                if categorical_values.nunique(dropna=True) > self.max_categories:
                    continue

                strength = self._eta_squared(
                    target_values if target_numeric else feature_values,
                    feature_values if target_numeric else target_values,
                )
                threshold = self.categorical_threshold
                metric = "eta_squared"
            else:
                if (
                    target_values.nunique(dropna=True) > self.max_categories
                    or feature_values.nunique(dropna=True) > self.max_categories
                ):
                    continue

                strength = self._normalized_mutual_information(
                    target_values,
                    feature_values,
                )
                threshold = self.categorical_threshold
                metric = "normalized_mutual_information"

            if pd.isna(strength) or strength < threshold:
                continue

            findings.append(
                Finding(
                    detector=self.name,
                    category="target_leakage",
                    severity="high",
                    confidence=float(strength),
                    explanation=(
                        f"Feature '{feature}' has a strong statistical "
                        f"association with target '{target}' "
                        f"({metric}={strength:.4f}). This may indicate "
                        "target-derived or post-outcome information."
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
                        "metric": metric,
                        "strength": float(strength),
                        "threshold": threshold,
                        "sample_count": len(pair),
                        "target_numeric": target_numeric,
                        "feature_numeric": feature_numeric,
                    },
                )
            )

        return findings

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        return bool(
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )

    @staticmethod
    def _pearson_strength(
        target: pd.Series,
        feature: pd.Series,
    ) -> float:
        correlation = target.corr(feature)
        if pd.isna(correlation):
            return float("nan")
        return abs(float(correlation))

    @staticmethod
    def _eta_squared(
        numeric: pd.Series,
        groups: pd.Series,
    ) -> float:
        numeric = pd.to_numeric(numeric, errors="coerce")
        pair = pd.DataFrame({"numeric": numeric, "groups": groups}).dropna()

        if len(pair) < 2:
            return float("nan")

        values = pair["numeric"]
        total_mean = float(values.mean())
        total_ss = float(((values - total_mean) ** 2).sum())

        if total_ss <= 0.0:
            return float("nan")

        group_stats = pair.groupby("groups", dropna=False)["numeric"].agg(
            ["mean", "size"]
        )
        between_ss = float(
            (
                group_stats["size"]
                * (group_stats["mean"] - total_mean) ** 2
            ).sum()
        )

        return min(1.0, max(0.0, between_ss / total_ss))

    @staticmethod
    def _normalized_mutual_information(
        target: pd.Series,
        feature: pd.Series,
    ) -> float:
        table = pd.crosstab(target, feature)
        if table.empty:
            return float("nan")

        counts = table.to_numpy(dtype=float)
        total = counts.sum()
        if total <= 0.0:
            return float("nan")

        probabilities = counts / total
        row_marginals = probabilities.sum(axis=1)
        column_marginals = probabilities.sum(axis=0)

        mutual_information = 0.0
        for row_index in range(probabilities.shape[0]):
            for column_index in range(probabilities.shape[1]):
                probability = probabilities[row_index, column_index]
                if probability <= 0.0:
                    continue
                denominator = (
                    row_marginals[row_index] * column_marginals[column_index]
                )
                if denominator <= 0.0:
                    continue
                mutual_information += probability * math.log(
                    probability / denominator
                )

        target_entropy = -sum(
            probability * math.log(probability)
            for probability in row_marginals
            if probability > 0.0
        )
        feature_entropy = -sum(
            probability * math.log(probability)
            for probability in column_marginals
            if probability > 0.0
        )

        denominator = math.sqrt(target_entropy * feature_entropy)
        if denominator <= 0.0:
            return float("nan")

        return min(1.0, max(0.0, mutual_information / denominator))
