from __future__ import annotations

import pandas as pd

from dataleaks.detectors.cross_dataset.overlap import (
    CrossDatasetOverlapDetector,
)
from dataleaks.detectors.feature.identifier import (
    IdentifierLeakageDetector,
)
from dataleaks.detectors.feature.suspicious import (
    SuspiciousFeatureDetector,
)
from dataleaks.detectors.feature.target_encoding import (
    TargetEncodingLeakageDetector,
)
from dataleaks.detectors.preprocessing.contamination import (
    PreprocessingContaminationDetector,
)
from dataleaks.detectors.preprocessing.fit_before_split import (
    FitBeforeSplitDetector,
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
from dataleaks.detectors.target.derived import (
    DerivedTargetLeakageDetector,
)
from dataleaks.detectors.target.direct import (
    DirectTargetLeakageDetector,
)
from dataleaks.detectors.target.statistical import (
    StatisticalTargetLeakageDetector,
)
from dataleaks.detectors.temporal.future_features import (
    FutureFeatureDetector,
)
from dataleaks.detectors.temporal.time_order import (
    TimeOrderDetector,
)
from dataleaks.engine.registry import DetectorRegistry
from dataleaks.schemas.config import DataLeaksConfig
from dataleaks.schemas.dataset import DatasetContext


def build_default_registry(
    context: DatasetContext,
    config: DataLeaksConfig,
) -> DetectorRegistry:
    """Build the default detector registry for a dataset context."""

    if not isinstance(context, DatasetContext):
        raise TypeError("context must be a DatasetContext")

    if not isinstance(config, DataLeaksConfig):
        raise TypeError("config must be a DataLeaksConfig")

    registry = DetectorRegistry()

    # Target leakage detectors
    if config.enable_target_checks:
        registry.register(DirectTargetLeakageDetector)
        registry.register(StatisticalTargetLeakageDetector)
        registry.register(DerivedTargetLeakageDetector)
    else:
        _skip_category(
            registry,
            [
                DirectTargetLeakageDetector,
                StatisticalTargetLeakageDetector,
                DerivedTargetLeakageDetector,
            ],
            reason="disabled_by_config",
        )

    # Split leakage detectors
    if config.enable_split_checks:
        registry.register(SplitDuplicateDetector)

        registry.register(
            _build_near_duplicate_detector(config)
        )

        registry.register(SplitOverlapDetector)
    else:
        _skip_category(
            registry,
            [
                SplitDuplicateDetector,
                NearDuplicateDetector,
                SplitOverlapDetector,
            ],
            reason="disabled_by_config",
        )

    # Temporal leakage detectors
    if config.enable_temporal_checks:
        _register_temporal_detectors(
            registry,
            context,
        )
    else:
        _skip_category(
            registry,
            [
                FutureFeatureDetector,
                TimeOrderDetector,
            ],
            reason="disabled_by_config",
        )

    # Preprocessing leakage detectors
    if config.enable_preprocessing_checks:
        registry.register(FitBeforeSplitDetector)
        registry.register(PreprocessingContaminationDetector)
    else:
        _skip_category(
            registry,
            [
                FitBeforeSplitDetector,
                PreprocessingContaminationDetector,
            ],
            reason="disabled_by_config",
        )

    # Feature leakage detectors
    if config.enable_feature_checks:
        registry.register(TargetEncodingLeakageDetector)
        registry.register(IdentifierLeakageDetector)
        registry.register(SuspiciousFeatureDetector)
    else:
        _skip_category(
            registry,
            [
                TargetEncodingLeakageDetector,
                IdentifierLeakageDetector,
                SuspiciousFeatureDetector,
            ],
            reason="disabled_by_config",
        )

    # Cross-dataset leakage detector
    if config.enable_cross_dataset_checks:
        _register_cross_dataset_detector(
            registry,
            context,
        )
    else:
        _skip_category(
            registry,
            [
                CrossDatasetOverlapDetector,
            ],
            reason="disabled_by_config",
        )

    return registry


def _build_near_duplicate_detector(
    config: DataLeaksConfig,
) -> type[NearDuplicateDetector]:
    """Create a configured near-duplicate detector class."""

    threshold = config.near_duplicate_threshold

    class ConfiguredNearDuplicateDetector(
        NearDuplicateDetector
    ):
        def __init__(self) -> None:
            super().__init__(
                threshold=threshold,
            )

    ConfiguredNearDuplicateDetector.name = (
        NearDuplicateDetector.name
    )
    ConfiguredNearDuplicateDetector.category = (
        NearDuplicateDetector.category
    )

    return ConfiguredNearDuplicateDetector


def _register_temporal_detectors(
    registry: DetectorRegistry,
    context: DatasetContext,
) -> None:
    """Register temporal detectors when their configuration exists."""

    temporal = context.metadata.get("temporal")

    if temporal is None:
        return

    if not isinstance(temporal, dict):
        raise TypeError(
            "context.metadata['temporal'] must be a dictionary"
        )

    prediction_time_column = temporal.get(
        "prediction_time_column"
    )

    feature_time_columns = temporal.get(
        "feature_time_columns"
    )

    if (
        prediction_time_column is not None
        and feature_time_columns is not None
    ):
        if not isinstance(prediction_time_column, str):
            raise TypeError(
                "temporal 'prediction_time_column' "
                "must be a string"
            )

        if not isinstance(feature_time_columns, list):
            raise TypeError(
                "temporal 'feature_time_columns' "
                "must be a list"
            )

        class ConfiguredFutureFeatureDetector(
            FutureFeatureDetector
        ):
            def __init__(self) -> None:
                super().__init__(
                    prediction_time_column=prediction_time_column,
                    feature_time_columns=feature_time_columns,
                )

        ConfiguredFutureFeatureDetector.name = (
            FutureFeatureDetector.name
        )
        ConfiguredFutureFeatureDetector.category = (
            FutureFeatureDetector.category
        )

        registry.register(
            ConfiguredFutureFeatureDetector
        )

    time_column = temporal.get("time_column")

    if time_column is not None:
        if not isinstance(time_column, str):
            raise TypeError(
                "temporal 'time_column' must be a string"
            )

        class ConfiguredTimeOrderDetector(
            TimeOrderDetector
        ):
            def __init__(self) -> None:
                super().__init__(
                    time_column=time_column,
                )

        ConfiguredTimeOrderDetector.name = (
            TimeOrderDetector.name
        )
        ConfiguredTimeOrderDetector.category = (
            TimeOrderDetector.category
        )

        registry.register(
            ConfiguredTimeOrderDetector
        )


def _register_cross_dataset_detector(
    registry: DetectorRegistry,
    context: DatasetContext,
) -> None:
    """Register cross-dataset detection when reference data exists."""

    reference_data = context.metadata.get(
        "reference_data"
    )

    if reference_data is None:
        return

    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError(
            "context.metadata['reference_data'] "
            "must be a pandas DataFrame"
        )

    columns = context.metadata.get(
        "reference_columns"
    )

    if columns is not None and not isinstance(columns, list):
        raise TypeError(
            "context.metadata['reference_columns'] "
            "must be a list"
        )

    class ConfiguredCrossDatasetOverlapDetector(
        CrossDatasetOverlapDetector
    ):
        def __init__(self) -> None:
            super().__init__(
                reference_data=reference_data,
                columns=columns,
            )

    ConfiguredCrossDatasetOverlapDetector.name = (
        CrossDatasetOverlapDetector.name
    )
    ConfiguredCrossDatasetOverlapDetector.category = (
        CrossDatasetOverlapDetector.category
    )

    registry.register(
        ConfiguredCrossDatasetOverlapDetector
    )


def _skip_category(
    registry: DetectorRegistry,
    detectors: list[type],
    *,
    reason: str,
) -> None:
    """Mark a detector category as intentionally skipped."""

    for detector in detectors:
        registry.skip(
            detector,
            reason=reason,
        )