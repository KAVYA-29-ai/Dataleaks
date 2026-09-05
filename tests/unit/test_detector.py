import pandas as pd
import pytest

from dataleaks.engine.detector import BaseDetector
from dataleaks.schemas.dataset import DatasetContext


class ExampleDetector(BaseDetector):
    name = "example"
    category = "test"

    def detect(self, context):
        return []


def test_detector_contract():
    detector = ExampleDetector()

    context = DatasetContext(
        data=pd.DataFrame({"target": [0, 1, 0]}),
        target="target",
    )

    result = detector.detect(context)

    assert detector.name == "example"
    assert detector.category == "test"
    assert result == []


def test_base_detector_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseDetector()