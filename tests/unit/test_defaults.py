import pandas as pd
import pytest

from dataleaks.detectors.target.direct import (
    DirectTargetLeakageDetector,
)
from dataleaks.detectors.target.statistical import (
    StatisticalTargetLeakageDetector,
)
from dataleaks.detectors.target.derived import (
    DerivedTargetLeakageDetector,
)
from dataleaks.detectors.split.duplicates import (
    SplitDuplicateDetector,
)
from dataleaks.detectors.split.near_duplicates import (
    NearDuplicateDetector,
)
from dataleaks.detectors.split.overlap import (
    SplitOverlapDetector,
)
from dataleaks.detectors.preprocessing.fit_before_split import (
    FitBeforeSplitDetector,
)
from dataleaks.detectors.preprocessing.contamination import (
    PreprocessingContaminationDetector,
)
from dataleaks.detectors.feature.target_encoding import (
    TargetEncodingLeakageDetector,
)
from dataleaks.detectors.feature.identifier import (
    IdentifierLeakageDetector,
)
from dataleaks.detectors.feature.suspicious import (
    SuspiciousFeatureDetector,
)
from dataleaks.detectors.temporal.future_features import (
    FutureFeatureDetector,
)
from dataleaks.detectors.temporal.time_order import (
    TimeOrderDetector,
)
from dataleaks.detectors.cross_dataset.overlap import (
    CrossDatasetOverlapDetector,
)
from dataleaks.engine.defaults import build_default_registry
from dataleaks.schemas.config import DataLeaksConfig
from dataleaks.schemas.dataset import DatasetContext


def make_context(
    metadata=None,
    train=None,
    test=None,
):
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    return DatasetContext(
        data=data,
        target="target",
        train=train,
        test=test,
        metadata=metadata or {},
    )


def test_default_registry_registers_target_detectors():
    context = make_context()

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "target_direct" in registry.names()
    assert "target_statistical" in registry.names()
    assert "target_derived" in registry.names()


def test_default_registry_registers_split_detectors():
    context = make_context()

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "split_duplicates" in registry.names()
    assert "split_near_duplicates" in registry.names()
    assert "split_overlap" in registry.names()


def test_default_registry_registers_preprocessing_detectors():
    context = make_context()

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "preprocessing_fit_before_split" in registry.names()
    assert "preprocessing_contamination" in registry.names()


def test_default_registry_registers_feature_detectors():
    context = make_context()

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "feature_target_encoding" in registry.names()
    assert "feature_identifier" in registry.names()
    assert "feature_suspicious" in registry.names()


def test_temporal_detectors_are_not_registered_without_config():
    context = make_context()

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "temporal_future_features" not in registry.names()
    assert "temporal_time_order" not in registry.names()


def test_cross_dataset_detector_is_not_registered_without_reference():
    context = make_context()

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "cross_dataset_overlap" not in registry.names()


def test_temporal_detectors_are_registered_with_config():
    context = make_context(
        metadata={
            "temporal": {
                "prediction_time_column": "prediction_time",
                "feature_time_columns": ["feature_time"],
                "time_column": "timestamp",
            }
        }
    )

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "temporal_future_features" in registry.names()
    assert "temporal_time_order" in registry.names()


def test_cross_dataset_detector_is_registered_with_reference():
    reference = pd.DataFrame(
        {
            "feature": [10, 20],
            "target": [0, 1],
        }
    )

    context = make_context(
        metadata={
            "reference_data": reference,
        }
    )

    registry = build_default_registry(
        context,
        DataLeaksConfig(),
    )

    assert "cross_dataset_overlap" in registry.names()


def test_target_checks_can_be_disabled():
    context = make_context()

    config = DataLeaksConfig(
        enable_target_checks=False,
    )

    registry = build_default_registry(
        context,
        config,
    )

    assert "target_direct" not in registry.names()
    assert "target_statistical" not in registry.names()
    assert "target_derived" not in registry.names()


def test_split_checks_can_be_disabled():
    context = make_context()

    config = DataLeaksConfig(
        enable_split_checks=False,
    )

    registry = build_default_registry(
        context,
        config,
    )

    assert "split_duplicates" not in registry.names()
    assert "split_near_duplicates" not in registry.names()
    assert "split_overlap" not in registry.names()


def test_temporal_checks_can_be_disabled():
    context = make_context(
        metadata={
            "temporal": {
                "prediction_time_column": "prediction_time",
                "feature_time_columns": ["feature_time"],
                "time_column": "timestamp",
            }
        }
    )

    config = DataLeaksConfig(
        enable_temporal_checks=False,
    )

    registry = build_default_registry(
        context,
        config,
    )

    assert "temporal_future_features" not in registry.names()
    assert "temporal_time_order" not in registry.names()


def test_preprocessing_checks_can_be_disabled():
    context = make_context()

    config = DataLeaksConfig(
        enable_preprocessing_checks=False,
    )

    registry = build_default_registry(
        context,
        config,
    )

    assert "preprocessing_fit_before_split" not in registry.names()
    assert "preprocessing_contamination" not in registry.names()


def test_feature_checks_can_be_disabled():
    context = make_context()

    config = DataLeaksConfig(
        enable_feature_checks=False,
    )

    registry = build_default_registry(
        context,
        config,
    )

    assert "feature_target_encoding" not in registry.names()
    assert "feature_identifier" not in registry.names()
    assert "feature_suspicious" not in registry.names()


def test_cross_dataset_checks_can_be_disabled():
    reference = pd.DataFrame(
        {
            "feature": [10],
        }
    )

    context = make_context(
        metadata={
            "reference_data": reference,
        }
    )

    config = DataLeaksConfig(
        enable_cross_dataset_checks=False,
    )

    registry = build_default_registry(
        context,
        config,
    )

    assert "cross_dataset_overlap" not in registry.names()


def test_near_duplicate_threshold_is_passed_to_detector():
    context = make_context()

    config = DataLeaksConfig(
        near_duplicate_threshold=0.88,
    )

    registry = build_default_registry(
        context,
        config,
    )

    detector = registry.create_all()

    near_duplicate = next(
        item
        for item in detector
        if item.name == "split_near_duplicates"
    )

    assert near_duplicate.threshold == 0.88


def test_invalid_context_is_rejected():
    with pytest.raises(TypeError):
        build_default_registry(
            "not a context",
            DataLeaksConfig(),
        )


def test_invalid_config_is_rejected():
    context = make_context()

    with pytest.raises(TypeError):
        build_default_registry(
            context,
            "not a config",
        )


def test_invalid_temporal_metadata_is_rejected():
    context = make_context(
        metadata={
            "temporal": "invalid",
        }
    )

    with pytest.raises(TypeError):
        build_default_registry(
            context,
            DataLeaksConfig(),
        )


def test_invalid_reference_data_is_rejected():
    context = make_context(
        metadata={
            "reference_data": "not a dataframe",
        }
    )

    with pytest.raises(TypeError):
        build_default_registry(
            context,
            DataLeaksConfig(),
        )