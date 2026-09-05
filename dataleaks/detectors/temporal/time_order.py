from __future__ import annotations

import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class TimeOrderDetector(BaseDetector):
    """Detect chronological violations between train and test splits."""

    name = "temporal_time_order"
    category = "temporal_leakage"

    def __init__(self, time_column: str) -> None:
        if not time_column.strip():
            raise ValueError("time_column must not be empty")

        self.time_column = time_column

    def detect(self, context: DatasetContext) -> list[Finding]:
        train = context.train
        test = context.test

        if train is None or test is None:
            return []

        if train.empty or test.empty:
            return []

        if self.time_column not in train.columns:
            raise ValueError(
                f"Time column '{self.time_column}' is missing from train"
            )

        if self.time_column not in test.columns:
            raise ValueError(
                f"Time column '{self.time_column}' is missing from test"
            )

        train_time = pd.to_datetime(
            train[self.time_column],
            errors="coerce",
            format="mixed",
            utc=True,
        )

        test_time = pd.to_datetime(
            test[self.time_column],
            errors="coerce",
            format="mixed",
            utc=True,
        )

        train_valid = train_time.dropna()
        test_valid = test_time.dropna()

        if train_valid.empty or test_valid.empty:
            return []

        train_max = train_valid.max()
        test_min = test_valid.min()

        if train_max <= test_min:
            return []

        overlap_duration = train_max - test_min

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity="high",
                confidence=1.0,
                explanation=(
                    f"Training data extends to {train_max}, while test "
                    f"data begins at {test_min}. The chronological "
                    "ordering is violated."
                ),
                recommendation=(
                    "For forecasting or time-dependent evaluation, "
                    "ensure training observations occur before test "
                    "observations."
                ),
                affected_columns=[self.time_column],
                evidence={
                    "type": "chronological_split_violation",
                    "time_column": self.time_column,
                    "train_max": str(train_max),
                    "test_min": str(test_min),
                    "overlap_duration": str(overlap_duration),
                },
            )
        ]