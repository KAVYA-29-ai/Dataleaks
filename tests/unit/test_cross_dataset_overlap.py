import pandas as pd
import pytest

from dataleaks.detectors.cross_dataset.overlap import (
    CrossDatasetOverlapDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(data):
    return DatasetContext(data=data)


def test_detects_exact_cross_dataset_overlap():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [3, 4, 5],
            "value": [30, 40, 50],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].detector == "cross_dataset_overlap"
    assert findings[0].category == "cross_dataset_leakage"
    assert findings[0].evidence["overlap_count"] == 2


def test_no_overlap_produces_no_findings():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [4, 5, 6],
            "value": [40, 50, 60],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert findings == []


def test_custom_columns_are_supported():
    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "value": [100, 200, 300],
            "source": ["a", "b", "c"],
        }
    )

    reference = pd.DataFrame(
        {
            "customer_id": [3, 4],
            "value": [999, 400],
            "source": ["x", "y"],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["customer_id"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1
    assert findings[0].affected_columns == ["customer_id"]


def test_full_row_overlap_requires_matching_values():
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [2, 3],
            "value": [999, 30],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert findings == []


def test_duplicate_rows_are_counted_once():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "value": [10, 10, 20, 30],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 1, 4],
            "value": [10, 10, 40],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1


def test_high_overlap_gets_critical_severity():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 2, 3, 5],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["id"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].confidence == 0.75


def test_medium_overlap_gets_high_severity():
    data = pd.DataFrame(
        {
            "id": list(range(10)),
        }
    )

    reference = pd.DataFrame(
        {
            "id": [0, 1, 20, 21, 22],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["id"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].confidence == 0.2


def test_low_overlap_gets_medium_severity():
    data = pd.DataFrame(
        {
            "id": list(range(20)),
        }
    )

    reference = pd.DataFrame(
        {
            "id": [0, 100, 101],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["id"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].severity == "medium"
    assert findings[0].confidence == 0.05


def test_empty_current_dataset_is_clean():
    data = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 2],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert findings == []


def test_empty_reference_dataset_is_clean():
    data = pd.DataFrame(
        {
            "id": [1, 2],
        }
    )

    reference = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert findings == []


def test_missing_configured_column_raises_error():
    data = pd.DataFrame(
        {
            "id": [1, 2],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 2],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["customer_id"],
    )

    with pytest.raises(ValueError):
        detector.detect(make_context(data))


def test_empty_columns_configuration_is_rejected():
    reference = pd.DataFrame(
        {
            "id": [1, 2],
        }
    )

    with pytest.raises(ValueError):
        CrossDatasetOverlapDetector(
            reference_data=reference,
            columns=[],
        )


def test_invalid_reference_data_is_rejected():
    with pytest.raises(TypeError):
        CrossDatasetOverlapDetector(
            reference_data=[1, 2, 3],
        )


def test_evidence_contains_columns():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [2, 4],
        }
    )

    detector = CrossDatasetOverlapDetector(
        reference_data=reference,
    )

    findings = detector.detect(make_context(data))

    assert findings[0].evidence["columns"] == ["id"]