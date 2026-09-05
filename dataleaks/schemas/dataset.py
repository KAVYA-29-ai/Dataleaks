from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetContext:
    """Context shared across DataLeaks detection components."""

    data: pd.DataFrame
    target: str | None = None

    train: pd.DataFrame | None = None
    validation: pd.DataFrame | None = None
    test: pd.DataFrame | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    schema_validation: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if self.target is not None and self.target not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target}' does not exist in the dataset"
            )

        self.schema_validation = self._validate_schema()

    def _validate_schema(self) -> dict[str, Any]:
        """Validate train/test schema without blocking execution."""

        result: dict[str, Any] = {
            "schema_mismatch": False,
            "missing_from_train": [],
            "missing_from_test": [],
            "dtype_mismatches": [],
            "target": {
                "train_present": True,
                "test_present": True,
            },
        }

        if self.train is None or self.test is None:
            return result

        train_columns = set(self.train.columns)
        test_columns = set(self.test.columns)

        missing_from_train = sorted(test_columns - train_columns)
        missing_from_test = sorted(train_columns - test_columns)

        result["missing_from_train"] = missing_from_train
        result["missing_from_test"] = missing_from_test

        common_columns = sorted(train_columns & test_columns)

        for column in common_columns:
            train_dtype = str(self.train[column].dtype)
            test_dtype = str(self.test[column].dtype)

            if train_dtype != test_dtype:
                result["dtype_mismatches"].append(
                    {
                        "column": column,
                        "train_dtype": train_dtype,
                        "test_dtype": test_dtype,
                    }
                )

        target = self.target

        if target is not None:
            train_target_present = target in train_columns
            test_target_present = target in test_columns

            result["target"] = {
                "train_present": train_target_present,
                "test_present": test_target_present,
            }

        result["schema_mismatch"] = bool(
            missing_from_train
            or missing_from_test
            or result["dtype_mismatches"]
            or not result["target"]["train_present"]
            or not result["target"]["test_present"]
        )

        return result