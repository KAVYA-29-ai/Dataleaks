from __future__ import annotations

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class NearDuplicateDetector(BaseDetector):
    """Detect highly similar rows across training and test splits."""

    name = "split_near_duplicates"
    category = "split_leakage"

    def __init__(
        self,
        threshold: float = 0.95,
        min_comparable_columns: int = 2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        if min_comparable_columns < 1:
            raise ValueError("min_comparable_columns must be at least 1")

        self.threshold = threshold
        self.min_comparable_columns = min_comparable_columns

    def detect(self, context: DatasetContext) -> list[Finding]:
        train = context.train
        test = context.test

        if train is None or test is None:
            return []

        if train.empty or test.empty:
            return []

        common_columns = [
            column for column in train.columns if column in test.columns
        ]

        if len(common_columns) < self.min_comparable_columns:
            return []

        train_rows = train[common_columns].reset_index(drop=True)
        test_rows = test[common_columns].reset_index(drop=True)

        matches: list[dict[str, object]] = []

        for test_index, test_row in test_rows.iterrows():
            best_similarity = 0.0
            best_train_index: int | None = None
            best_matches = 0
            best_comparable = 0

            for train_index, train_row in train_rows.iterrows():
                similarity, matching, comparable = self._row_similarity(
                    train_row,
                    test_row,
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_train_index = train_index
                    best_matches = matching
                    best_comparable = comparable

            if (
                best_train_index is not None
                and best_similarity >= self.threshold
            ):
                matches.append(
                    {
                        "train_index": int(best_train_index),
                        "test_index": int(test_index),
                        "similarity": best_similarity,
                        "matching_columns": best_matches,
                        "comparable_columns": best_comparable,
                    }
                )

        if not matches:
            return []

        strongest_similarity = max(
            float(match["similarity"]) for match in matches
        )

        affected_columns = common_columns

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity="high",
                confidence=strongest_similarity,
                explanation=(
                    f"{len(matches)} test row(s) have near-duplicate "
                    f"matches in the training data at or above the "
                    f"{self.threshold:.2f} similarity threshold."
                ),
                recommendation=(
                    "Review near-duplicate samples across train and test "
                    "and remove duplicated entities or observations before "
                    "model evaluation."
                ),
                affected_columns=affected_columns,
                evidence={
                    "type": "near_duplicate_rows",
                    "matches": matches,
                    "threshold": self.threshold,
                    "common_columns": common_columns,
                },
            )
        ]

    @staticmethod
    def _row_similarity(
        train_row: pd.Series,
        test_row: pd.Series,
    ) -> tuple[float, int, int]:
        matching = 0
        comparable = 0

        for column in train_row.index:
            train_value = train_row[column]
            test_value = test_row[column]

            if pd.isna(train_value) and pd.isna(test_value):
                continue

            if pd.isna(train_value) or pd.isna(test_value):
                comparable += 1
                continue

            comparable += 1

            if train_value == test_value:
                matching += 1

        if comparable == 0:
            return 0.0, matching, comparable

        return matching / comparable, matching, comparable