import pandas as pd
import pytest

from dataleaks.schemas.dataset import DatasetContext


def test_dataset_context_creation():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [30000, 50000, 70000],
            "target": [0, 1, 1],
        }
    )

    context = DatasetContext(
        data=df,
        target="target",
    )

    assert context.data.equals(df)
    assert context.target == "target"


def test_invalid_target():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "target": [0, 1, 1],
        }
    )

    with pytest.raises(ValueError):
        DatasetContext(
            data=df,
            target="missing_target",
        )


def test_data_must_be_dataframe():
    with pytest.raises(TypeError):
        DatasetContext(
            data=[1, 2, 3],
            target=None,
        )