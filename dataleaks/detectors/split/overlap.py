from __future__ import annotations

import re

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class SplitOverlapDetector(BaseDetector):
    """Detect entity/key overlap between training and test splits."""

    name = "split_overlap"
    category = "split_leakage"

    # Automatic overlap detection should focus on columns that are
    # sufficiently likely to represent entities or keys.
    MIN_UNIQUE_RATIO = 0.95

    # Evidence must remain bounded regardless of dataset size.
    MAX_EVIDENCE_VALUES = 20

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

    def __init__(self, columns: list[str] | None = None) -> None:
        if columns is not None and not columns:
            raise ValueError(
                "columns must contain at least one column"
            )

        self.columns = columns

    def detect(
        self,
        context: DatasetContext,
    ) -> list[Finding]:
        train = context.train
        test = context.test

        if train is None or test is None:
            return []

        if train.empty or test.empty:
            return []

        columns = self._resolve_columns(
            train,
            test,
            target=context.target,
        )

        if not columns:
            return []

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

        findings: list[Finding] = []

        for column in columns:
            train_values = train[column].dropna()
            test_values = test[column].dropna()

            if train_values.empty or test_values.empty:
                continue

            train_unique = set(
                train_values.tolist()
            )
            test_unique = set(
                test_values.tolist()
            )

            if self.columns is None:
                if not self._is_entity_candidate(
                    column=column,
                    train_unique_count=len(train_unique),
                    test_unique_count=len(test_unique),
                    train_size=len(train_values),
                    test_size=len(test_values),
                    explicit_identifiers=explicit_identifiers,
                ):
                    continue

            overlap = train_unique.intersection(
                test_unique
            )

            if not overlap:
                continue

            overlap_count = len(overlap)
            test_unique_count = len(test_unique)

            overlap_ratio = (
                overlap_count / test_unique_count
                if test_unique_count
                else 0.0
            )

            if overlap_ratio >= 0.5:
                severity = "high"
            elif overlap_ratio >= 0.1:
                severity = "medium"
            else:
                severity = "low"

            name_matches_identifier = bool(
                self._IDENTIFIER_PATTERN.search(column)
            )

            explicitly_configured = (
                column in explicit_identifiers
            )

            bounded_overlap_values = sorted(
                overlap,
                key=lambda value: str(value),
            )[: self.MAX_EVIDENCE_VALUES]

            findings.append(
                Finding(
                    detector=self.name,
                    category=self.category,
                    severity=severity,
                    confidence=overlap_ratio,
                    explanation=(
                        f"Column '{column}' contains "
                        f"{overlap_count} entity value(s) "
                        "shared between training and test "
                        "datasets."
                    ),
                    recommendation=(
                        f"Verify that '{column}' represents "
                        "an entity or group that should remain "
                        "isolated between splits."
                    ),
                    affected_columns=[column],
                    evidence={
                        "type": "entity_overlap",
                        "column": column,
                        "overlap_count": overlap_count,
                        "train_unique_values": len(train_unique),
                        "test_unique_values": test_unique_count,
                        "overlap_ratio_in_test": overlap_ratio,
                        "overlap_values": bounded_overlap_values,
                        "name_matches_identifier_pattern": (
                            name_matches_identifier
                        ),
                        "explicit_identifier": (
                            explicitly_configured
                        ),
                        "unique_ratio_threshold": (
                            self.MIN_UNIQUE_RATIO
                        ),
                        "evidence_values_limit": (
                            self.MAX_EVIDENCE_VALUES
                        ),
                    },
                )
            )

        return findings

    def _resolve_columns(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str | None = None,
    ) -> list[str]:
        if self.columns is not None:
            missing = [
                column
                for column in self.columns
                if (
                    column not in train.columns
                    or column not in test.columns
                )
            ]

            if missing:
                raise ValueError(
                    "Configured overlap columns are missing: "
                    f"{missing}"
                )

            return [
                column
                for column in self.columns
                if column != target
            ]

        return [
            column
            for column in train.columns
            if (
                column in test.columns
                and column != target
            )
        ]

    def _is_entity_candidate(
        self,
        *,
        column: str,
        train_unique_count: int,
        test_unique_count: int,
        train_size: int,
        test_size: int,
        explicit_identifiers: list[str],
    ) -> bool:
        """Determine whether an automatic column is entity-like."""

        if train_size <= 0 or test_size <= 0:
            return False

        if column in explicit_identifiers:
            return True

        if self._IDENTIFIER_PATTERN.search(column):
            return True

        train_unique_ratio = (
            train_unique_count / train_size
        )
        test_unique_ratio = (
            test_unique_count / test_size
        )

        return (
            train_unique_ratio >= self.MIN_UNIQUE_RATIO
            and test_unique_ratio >= self.MIN_UNIQUE_RATIO
        )