from __future__ import annotations

import re
from typing import Any

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class IdentifierLeakageDetector(BaseDetector):
    """Detect identifier-like features that may enable entity memorization."""

    name = "feature_identifier"
    category = "feature_leakage"

    MAX_EVIDENCE_ROWS = 20

    _IDENTIFIER_PATTERN = re.compile(
        r"(^|_)("
        r"id|ids|uuid|guid|"
        r"user_id|customer_id|client_id|"
        r"account_id|member_id|"
        r"patient_id|transaction_id|"
        r"record_id|entity_id"
        r")(_|$)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        uniqueness_threshold: float = 0.95,
    ) -> None:
        if not 0.0 <= uniqueness_threshold <= 1.0:
            raise ValueError(
                "uniqueness_threshold must be between 0.0 and 1.0"
            )

        self.uniqueness_threshold = uniqueness_threshold

    @staticmethod
    def _safe_value(value: Any) -> str | None:
        """Return a bounded representation for evidence."""

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        value = str(value)

        if len(value) > 64:
            return value[:61] + "..."

        return value

    def _build_identifier_evidence(
        self,
        context: DatasetContext,
        column: str,
    ) -> list[dict[str, object]]:
        """Build bounded row-level evidence."""

        evidence: list[dict[str, object]] = []

        for row_index, value in context.data[column].items():
            if pd.isna(value):
                continue

            evidence.append(
                {
                    "row_index": row_index,
                    "identifier_value": self._safe_value(
                        value
                    ),
                }
            )

            if len(evidence) >= self.MAX_EVIDENCE_ROWS:
                break

        return evidence

    def detect(
        self,
        context: DatasetContext,
    ) -> list[Finding]:
        findings: list[Finding] = []

        explicit_identifiers = context.metadata.get(
            "identifier_columns",
            [],
        )

        if explicit_identifiers is None:
            explicit_identifiers = []

        if not isinstance(explicit_identifiers, list):
            raise TypeError(
                "context.metadata['identifier_columns'] must be a list"
            )

        for column in context.data.columns:
            if column == context.target:
                continue

            series = context.data[column]

            if series.empty:
                continue

            non_null = series.dropna()

            if non_null.empty:
                continue

            uniqueness_ratio = (
                non_null.nunique(dropna=True)
                / len(non_null)
            )

            name_match = bool(
                self._IDENTIFIER_PATTERN.search(column)
            )

            explicit_match = (
                column in explicit_identifiers
            )

            # High cardinality alone is not enough.
            if not name_match and not explicit_match:
                continue

            overlap_count = 0

            if (
                context.train is not None
                and context.test is not None
                and column in context.train.columns
                and column in context.test.columns
            ):
                train_values = set(
                    context.train[column]
                    .dropna()
                    .tolist()
                )

                test_values = set(
                    context.test[column]
                    .dropna()
                    .tolist()
                )

                overlap_count = len(
                    train_values.intersection(
                        test_values
                    )
                )

            if overlap_count > 0:
                severity = "high"

                confidence = min(
                    1.0,
                    max(
                        uniqueness_ratio,
                        0.5,
                    ),
                )

            elif uniqueness_ratio >= self.uniqueness_threshold:
                severity = "medium"
                confidence = uniqueness_ratio

            else:
                severity = "low"
                confidence = uniqueness_ratio

            explanation = (
                f"Feature '{column}' appears to be identifier-like "
                f"with a uniqueness ratio of "
                f"{uniqueness_ratio:.4f}."
            )

            if overlap_count > 0:
                explanation += (
                    f" {overlap_count} identifier value(s) are shared "
                    "between training and test data."
                )

            recommendation = (
                f"Review whether '{column}' should be available to "
                "the model. If it identifies entities, ensure "
                "entity-level separation between training and test "
                "data."
            )

            evidence = {
                "type": "identifier_like_feature",
                "column": column,
                "uniqueness_ratio": uniqueness_ratio,
                "name_matches_identifier_pattern": name_match,
                "explicit_identifier": explicit_match,
                "train_test_overlap_count": overlap_count,
                "uniqueness_threshold": (
                    self.uniqueness_threshold
                ),
                "evidence_limit": (
                    self.MAX_EVIDENCE_ROWS
                ),
                "row_evidence": (
                    self._build_identifier_evidence(
                        context,
                        column,
                    )
                ),
            }

            findings.append(
                Finding(
                    detector=self.name,
                    category=self.category,
                    severity=severity,
                    confidence=confidence,
                    explanation=explanation,
                    recommendation=recommendation,
                    affected_columns=[column],
                    evidence=evidence,
                )
            )

        return findings