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

    MIN_UNIQUE_RATIO = 0.95
    MAX_EVIDENCE_VALUES = 20
    NUMERIC_LIKE_RATIO = 0.98

    _IDENTIFIER_PATTERN = re.compile(
        r"(^|_)("
        r"id|ids|uuid|guid|"
        r"user_id|customer_id|client_id|"
        r"account_id|member_id|patient_id|"
        r"transaction_id|record_id|entity_id|"
        r"user_number|customer_number|client_number|"
        r"account_number|member_number|transaction_number|"
        r"record_number|entity_number"
        r")(_|$)",
        re.IGNORECASE,
    )

    def __init__(self, columns: list[str] | None = None) -> None:
        if columns is not None and not columns:
            raise ValueError("columns must contain at least one column")

        self.columns = columns

    @staticmethod
    def _normalize_column_name(column: str) -> str:
        normalized = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            str(column),
        )
        normalized = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            normalized,
        )
        return normalized.strip("_").lower()

    @classmethod
    def _matches_identifier_pattern(cls, column: str) -> bool:
        normalized = cls._normalize_column_name(column)
        return bool(cls._IDENTIFIER_PATTERN.search(normalized))

    @classmethod
    def _looks_numeric(cls, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return True

        non_null = series.dropna()
        if non_null.empty:
            return False

        converted = pd.to_numeric(non_null, errors="coerce")
        return bool(converted.notna().mean() >= cls.NUMERIC_LIKE_RATIO)

    def detect(self, context: DatasetContext) -> list[Finding]:
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

            train_unique = set(train_values.tolist())
            test_unique = set(test_values.tolist())

            if self.columns is None and not self._is_entity_candidate(
                column=column,
                train_unique_count=len(train_unique),
                test_unique_count=len(test_unique),
                train_size=len(train_values),
                test_size=len(test_values),
                train_values=train_values,
                test_values=test_values,
                explicit_identifiers=explicit_identifiers,
            ):
                continue

            overlap = train_unique.intersection(test_unique)
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

            name_matches_identifier = self._matches_identifier_pattern(column)
            explicitly_configured = column in explicit_identifiers

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
                        f"Column '{column}' contains {overlap_count} entity "
                        "value(s) shared between training and test datasets."
                    ),
                    recommendation=(
                        f"Verify that '{column}' represents an entity or "
                        "group that should remain isolated between splits."
                    ),
                    affected_columns=[column],
                    evidence={
                        "type": "entity_overlap",
                        "column": column,
                        "normalized_column_name": self._normalize_column_name(
                            column
                        ),
                        "overlap_count": overlap_count,
                        "train_unique_values": len(train_unique),
                        "test_unique_values": test_unique_count,
                        "overlap_ratio_in_test": overlap_ratio,
                        "overlap_values": bounded_overlap_values,
                        "name_matches_identifier_pattern": name_matches_identifier,
                        "explicit_identifier": explicitly_configured,
                        "numeric_like": self._looks_numeric(train_values),
                        "unique_ratio_threshold": self.MIN_UNIQUE_RATIO,
                        "evidence_values_limit": self.MAX_EVIDENCE_VALUES,
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
                if column not in train.columns or column not in test.columns
            ]
            if missing:
                raise ValueError(
                    "Configured overlap columns are missing: "
                    f"{missing}"
                )

            return [column for column in self.columns if column != target]

        return [
            column
            for column in train.columns
            if column in test.columns and column != target
        ]

    def _is_entity_candidate(
        self,
        *,
        column: str,
        train_unique_count: int,
        test_unique_count: int,
        train_size: int,
        test_size: int,
        train_values: pd.Series,
        test_values: pd.Series,
        explicit_identifiers: list[str],
    ) -> bool:
        """Determine whether an automatic column is entity-like."""

        if train_size <= 0 or test_size <= 0:
            return False

        if column in explicit_identifiers:
            return True

        if self._matches_identifier_pattern(column):
            return True

        # Numeric measurements such as TotalCharges can naturally repeat
        # across train/test. High cardinality alone is not evidence that
        # such a column is an entity key.
        if self._looks_numeric(train_values) or self._looks_numeric(test_values):
            return False

        train_unique_ratio = train_unique_count / train_size
        test_unique_ratio = test_unique_count / test_size

        return (
            train_unique_ratio >= self.MIN_UNIQUE_RATIO
            and test_unique_ratio >= self.MIN_UNIQUE_RATIO
        )
