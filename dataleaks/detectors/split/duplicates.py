import pandas as pd

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext
from dataleaks.schemas.finding import Finding


class SplitDuplicateDetector(BaseDetector):
    """Detect exact row overlap between training and test datasets."""

    name = "split_duplicates"
    category = "split_leakage"

    def detect(self, context: DatasetContext) -> list[Finding]:
        train = context.train
        test = context.test

        if train is None or test is None:
            return []

        if train.empty or test.empty:
            return []

        common_columns = [column for column in train.columns if column in test.columns]

        if not common_columns:
            return []

        train_rows = train[common_columns].drop_duplicates()
        test_rows = test[common_columns].drop_duplicates()

        train_index = pd.MultiIndex.from_frame(train_rows.reset_index(drop=True))
        test_index = pd.MultiIndex.from_frame(test_rows.reset_index(drop=True))

        overlap = train_index.intersection(test_index)
        overlap_count = len(overlap)

        if overlap_count == 0:
            return []

        test_unique_count = len(test_rows)

        overlap_ratio = (
            overlap_count / test_unique_count
            if test_unique_count
            else 0.0
        )

        if overlap_ratio >= 0.5:
            severity = "critical"
        elif overlap_ratio >= 0.1:
            severity = "high"
        else:
            severity = "medium"

        confidence = min(1.0, overlap_ratio)

        return [
            Finding(
                detector=self.name,
                category=self.category,
                severity=severity,
                confidence=confidence,
                explanation=(
                    f"{overlap_count} exact row(s) are shared between "
                    "the training and test datasets."
                ),
                recommendation=(
                    "Remove overlapping samples and rebuild the split "
                    "before evaluating the model."
                ),
                affected_columns=common_columns,
                evidence={
                    "type": "exact_row_overlap",
                    "overlap_count": overlap_count,
                    "train_unique_rows": len(train_rows),
                    "test_unique_rows": test_unique_count,
                    "overlap_ratio_in_test": overlap_ratio,
                    "common_columns": common_columns,
                },
            )
        ]