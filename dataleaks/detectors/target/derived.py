from __future__ import annotations

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class DerivedTargetLeakageDetector(BaseDetector):
    """Detect deterministic target-derived feature transformations."""

    name = "target_derived"
    category = "target_leakage"

    MAX_CATEGORICAL_LEVELS = 20

    def __init__(self, tolerance: float = 1e-9) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.tolerance = tolerance

    def detect(self, context: DatasetContext) -> list[Finding]:
        if context.target is None:
            return []

        target_name = context.target
        data = context.data
        target = data[target_name]

        findings: list[Finding] = []

        for column in data.columns:
            if column == target_name:
                continue

            feature = data[column]
            pair = pd.concat([feature, target], axis=1).dropna()

            if len(pair) < 2:
                continue

            feature_values = pair.iloc[:, 0]
            target_values = pair.iloc[:, 1]

            if self._is_numeric(target_values):
                if not self._is_numeric(feature_values):
                    continue

                if self._is_constant(feature_values):
                    continue
                if self._is_constant(target_values):
                    continue

                if not self._is_affine_transform(
                    feature_values,
                    target_values,
                ):
                    continue

                scale, offset = self._fit_affine_transform(
                    feature_values,
                    target_values,
                )

                findings.append(
                    Finding(
                        detector=self.name,
                        category=self.category,
                        severity="critical",
                        confidence=1.0,
                        explanation=(
                            f"Feature '{column}' can be deterministically "
                            f"expressed as approximately "
                            f"{scale:.6g} * '{target_name}' + {offset:.6g}."
                        ),
                        recommendation=(
                            f"Remove '{column}' or verify that it is "
                            "available before the prediction target is "
                            "generated."
                        ),
                        affected_columns=[column],
                        evidence={
                            "type": "affine_target_transform",
                            "target": target_name,
                            "scale": scale,
                            "offset": offset,
                            "samples": len(pair),
                            "tolerance": self.tolerance,
                        },
                    )
                )
                continue

            if self._is_deterministic_categorical_encoding(
                target_values,
                feature_values,
            ):
                mapping = self._build_mapping(
                    target_values,
                    feature_values,
                )

                findings.append(
                    Finding(
                        detector=self.name,
                        category=self.category,
                        severity="critical",
                        confidence=1.0,
                        explanation=(
                            f"Feature '{column}' is a deterministic encoding "
                            f"of categorical target '{target_name}'."
                        ),
                        recommendation=(
                            f"Remove '{column}' or ensure its encoding is "
                            "created only after the prediction target is "
                            "known, not before model training."
                        ),
                        affected_columns=[column],
                        evidence={
                            "type": "categorical_target_encoding",
                            "target": target_name,
                            "target_unique_values": int(
                                target_values.nunique(dropna=True)
                            ),
                            "feature_unique_values": int(
                                feature_values.nunique(dropna=True)
                            ),
                            "mapping": mapping,
                            "samples": len(pair),
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

    def _is_deterministic_categorical_encoding(
        self,
        target: pd.Series,
        feature: pd.Series,
    ) -> bool:
        target_unique = target.nunique(dropna=True)
        feature_unique = feature.nunique(dropna=True)

        if target_unique < 2:
            return False

        if (
            target_unique > self.MAX_CATEGORICAL_LEVELS
            or feature_unique > self.MAX_CATEGORICAL_LEVELS
        ):
            return False

        table = pd.crosstab(target, feature)
        if table.empty:
            return False

        nonzero_per_target = (table > 0).sum(axis=1)
        nonzero_per_feature = (table > 0).sum(axis=0)

        return bool(
            (nonzero_per_target == 1).all()
            and (nonzero_per_feature == 1).all()
            and target_unique == feature_unique
        )

    @staticmethod
    def _build_mapping(
        target: pd.Series,
        feature: pd.Series,
    ) -> list[dict[str, str]]:
        mapping: list[dict[str, str]] = []

        table = pd.crosstab(target, feature)
        for target_value in table.index:
            feature_values = table.loc[target_value]
            feature_value = feature_values.idxmax()
            mapping.append(
                {
                    "target": str(target_value),
                    "feature": str(feature_value),
                }
            )

        return mapping

    def _is_affine_transform(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> bool:
        if self._is_constant(feature):
            return False

        if self._is_constant(target):
            return False

        scale, offset = self._fit_affine_transform(
            feature,
            target,
        )

        if abs(scale) <= self.tolerance:
            return False

        predicted = target * scale + offset

        return bool(
            (feature - predicted).abs().le(self.tolerance).all()
        )

    def _is_constant(
        self,
        series: pd.Series,
    ) -> bool:
        if series.empty:
            return True

        minimum = float(series.min())
        maximum = float(series.max())

        return abs(maximum - minimum) <= self.tolerance

    @staticmethod
    def _fit_affine_transform(
        feature: pd.Series,
        target: pd.Series,
    ) -> tuple[float, float]:
        target_mean = float(target.mean())
        feature_mean = float(feature.mean())
        centered_target = target - target_mean
        centered_feature = feature - feature_mean
        denominator = float((centered_target**2).sum())

        if denominator == 0.0:
            return 0.0, feature_mean

        scale = float(
            (centered_target * centered_feature).sum()
            / denominator
        )
        offset = feature_mean - scale * target_mean

        return scale, float(offset)
