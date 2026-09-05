import pandas as pd
import pytest

from dataleaks.detectors.cross_dataset.overlap import (
    CrossDatasetOverlapDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(data, target=None, metadata=None):
    return DatasetContext(
        data=data,
        target=target,
        metadata=metadata or {},
    )


def test_shared_target_values_alone_do_not_create_full_row_overlap():
    data = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    reference = pd.DataFrame(
        {
            "feature": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(
            data,
            target="target",
        )
    )

    assert findings == []


def test_single_common_low_cardinality_value_is_not_row_overlap():
    data = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D"],
            "value": [10, 20, 30, 40],
        }
    )

    reference = pd.DataFrame(
        {
            "category": ["A", "X", "Y", "Z"],
            "value": [999, 200, 300, 400],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert findings == []


def test_partial_column_match_does_not_count_as_row_overlap():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [2, 3, 4],
            "value": [999, 999, 40],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert findings == []


def test_exact_overlap_is_detected_automatically():
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

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert len(findings) == 1
    assert findings[0].detector == "cross_dataset_overlap"
    assert findings[0].category == "cross_dataset_leakage"
    assert findings[0].evidence["overlap_count"] == 2
    assert findings[0].evidence["overlap_ratio"] == 0.5


def test_duplicate_current_rows_do_not_inflate_overlap_ratio():
    data = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 3],
            "value": [10, 10, 10, 20, 30],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 4],
            "value": [10, 40],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1
    assert findings[0].evidence["current_unique_rows"] == 3
    assert findings[0].evidence["overlap_ratio"] == pytest.approx(1 / 3)


def test_nan_rows_are_handled_without_crashing():
    data = pd.DataFrame(
        {
            "id": [1, 2, None],
            "value": [10, None, 30],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 5, None],
            "value": [10, 50, 30],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 2


def test_configured_columns_only_control_overlap_matching():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 2, 4],
            "value": [999, 999, 40],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["id"],
    ).detect(
        make_context(data)
    )

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 2
    assert findings[0].affected_columns == ["id"]


def test_target_can_be_used_as_part_of_explicit_overlap_columns():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 4, 5],
            "target": [0, 9, 8],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
        columns=["id", "target"],
    ).detect(
        make_context(
            data,
            target="target",
        )
    )

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1


def test_automatic_mode_uses_only_common_columns():
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "value": [10, 20],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 3],
            "other_value": [10, 30],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1
    assert findings[0].affected_columns == ["id"]
    assert findings[0].evidence["columns"] == ["id"]


def test_reference_with_duplicate_rows_does_not_inflate_reference_count():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
        }
    )

    reference = pd.DataFrame(
        {
            "id": [1, 1, 1, 4],
        }
    )

    findings = CrossDatasetOverlapDetector(
        reference_data=reference,
    ).detect(
        make_context(data)
    )

    assert len(findings) == 1
    assert findings[0].evidence["overlap_count"] == 1
    assert findings[0].evidence["reference_unique_rows"] == 2