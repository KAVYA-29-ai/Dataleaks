from __future__ import annotations

import re

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class SuspiciousFeatureDetector(BaseDetector):
    """Detect features with unusually strong or semantically suspicious
    relationships with the target.
    """

    name = "feature_suspicious"
    category = "feature_leakage"

    _POST_OUTCOME_PATTERN = re.compile(
        r"("
        r"post[_-]?outcome|"
        r"after[_-]?outcome|"
        r"post[_-]?churn|"
        r"after[_-]?churn|"
        r"post[_-]?purchase|"
        r"after[_-]?purchase|"
        r"outcome[_-]?refund|"
        r"refund[_-]?after|"
        r"post[_-]?refund|"
        r"after[_-]?refund|"
        r"post[_-]?result|"
        r"after[_-]?result|"
        r"post[_-]?event|"
        r"after[_-]?event"
        r")",
        re.IGNORECASE,
    )

    _FUTURE_PATTERN = re.compile(
        r"("
        r"^future[_-]|"
        r"[_-]future$|"
        r"^next[_-]|"
        r"[_-]next$|"
        r"^upcoming[_-]|"
        r"[_-]upcoming$|"
        r"^subsequent[_-]|"
        r"[_-]subsequent$|"
        r"^projected[_-]|"
        r"[_-]projected$|"
        r"^forecast[_-]|"
        r"[_-]forecast$|"
        r"^post[_-]|"
        r"[_-]post$|"
        r"^after[_-]|"
        r"[_-]after$"
        r")",
        re.IGNORECASE,
    )

    def __init__(
        self,
        correlation_threshold: float = 0.999,
        min_samples: int = 10,
        post_outcome_correlation_threshold: float = 0.90,
        future_correlation_threshold: float = 0.90,
    ) -> None:
        if not 0.0 <= correlation_threshold <= 1.0:
            raise ValueError(
                "correlation_threshold must be between 0.0 and 1.0"
            )

        if min_samples < 2:
            raise ValueError(
                "min_samples must be at least 2"
            )

        if not 0.0 <= post_outcome_correlation_threshold <= 1.0:
            raise ValueError(
                "post_outcome_correlation_threshold must be "
                "between 0.0 and 1.0"
            )

        if not 0.0 <= future_correlation_threshold <= 1.0:
            raise ValueError(
                "future_correlation_threshold must be "
                "between 0.0 and 1.0"
            )

        self.correlation_threshold = correlation_threshold
        self.min_samples = min_samples
        self.post_outcome_correlation_threshold = (
            post_outcome_correlation_threshold
        )
        self.future_correlation_threshold = (
            future_correlation_threshold
        )

    def detect(
        self,
        context: DatasetContext,
    ) -> list[Finding]:
        if context.target is None:
            return []

        target = context.target
        data = context.data

        if target not in data.columns:
            return []

        if not pd.api.types.is_numeric_dtype(data[target]):
            return []

        findings: list[Finding] = []

        for column in data.columns:
            if column == target:
                continue

            feature = data[column]

            if not pd.api.types.is_numeric_dtype(feature):
                continue

            pair = pd.concat(
                [feature, data[target]],
                axis=1,
            ).dropna()

            if len(pair) < self.min_samples:
                continue

            feature_values = pair.iloc[:, 0]
            target_values = pair.iloc[:, 1]

            if feature_values.nunique(dropna=True) <= 1:
                continue

            if target_values.nunique(dropna=True) <= 1:
                continue

            correlation = feature_values.corr(target_values)

            if pd.isna(correlation):
                continue

            correlation = float(correlation)

            if abs(abs(correlation) - 1.0) <= 1e-12:
                correlation = 1.0 if correlation > 0 else -1.0
            else:
                correlation = max(-1.0, min(1.0, correlation))

            absolute_correlation = abs(correlation)

            is_post_outcome = bool(
                self._POST_OUTCOME_PATTERN.search(column)
            )

            is_future_semantic = bool(
                self._FUTURE_PATTERN.search(column)
            )

            standard_threshold_met = (
                absolute_correlation >= self.correlation_threshold
            )

            post_outcome_threshold_met = (
                is_post_outcome
                and absolute_correlation
                >= self.post_outcome_correlation_threshold
            )

            future_threshold_met = (
                is_future_semantic
                and absolute_correlation
                >= self.future_correlation_threshold
            )

            if not (
                standard_threshold_met
                or post_outcome_threshold_met
                or future_threshold_met
            ):
                continue

            if future_threshold_met:
                severity = "high"
                evidence_type = "future_semantic_feature"
                detection_reason = (
                    "future/post-event semantic feature name "
                    "combined with strong target relationship"
                )
            elif post_outcome_threshold_met:
                severity = "high"
                evidence_type = "post_outcome_feature"
                detection_reason = (
                    "post-outcome feature name combined "
                    "with strong target correlation"
                )
            else:
                severity = "high"
                evidence_type = "suspicious_target_relationship"
                detection_reason = (
                    "extremely strong target correlation"
                )

            semantic_warning = ""

            if future_threshold_met:
                semantic_warning = (
                    " Its name suggests that the value may "
                    "represent future, projected, or post-event "
                    "information."
                )
            elif is_post_outcome:
                semantic_warning = (
                    " Its name suggests that the feature may "
                    "contain post-outcome information."
                )

            findings.append(
                Finding(
                    detector=self.name,
                    category=self.category,
                    severity=severity,
                    confidence=absolute_correlation,
                    explanation=(
                        f"Feature '{column}' has a "
                        "suspiciously strong relationship "
                        f"with target '{target}' "
                        f"(|correlation|="
                        f"{absolute_correlation:.4f})."
                        f"{semantic_warning} "
                        "This may indicate target-derived "
                        "or future/post-outcome information."
                    ),
                    recommendation=(
                        f"Review how '{column}' is generated "
                        "and whether it is available before "
                        "prediction time. Remove it or rebuild "
                        "the feature using information available "
                        "at prediction time if it contains "
                        "future or target-derived information."
                    ),
                    affected_columns=[column],
                    evidence={
                        "type": evidence_type,
                        "feature": column,
                        "target": target,
                        "absolute_correlation": (
                            absolute_correlation
                        ),
                        "correlation": correlation,
                        "threshold": (
                            self.correlation_threshold
                        ),
                        "post_outcome_threshold": (
                            self.post_outcome_correlation_threshold
                        ),
                        "future_correlation_threshold": (
                            self.future_correlation_threshold
                        ),
                        "post_outcome_name_signal": (
                            is_post_outcome
                        ),
                        "future_semantic_name_signal": (
                            is_future_semantic
                        ),
                        "detection_reason": detection_reason,
                        "samples": len(pair),
                    },
                )
            )

        return findings