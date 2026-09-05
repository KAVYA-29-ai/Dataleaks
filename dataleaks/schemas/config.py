from dataclasses import dataclass, field


@dataclass
class DataLeaksConfig:
    """Configuration for DataLeaks detection."""

    enable_target_checks: bool = True
    enable_split_checks: bool = True
    enable_temporal_checks: bool = True
    enable_preprocessing_checks: bool = True
    enable_feature_checks: bool = True
    enable_cross_dataset_checks: bool = True

    duplicate_threshold: float = 0.0
    near_duplicate_threshold: float = 0.95

    confidence_threshold: float = 0.5

    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        thresholds = {
            "duplicate_threshold": self.duplicate_threshold,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "confidence_threshold": self.confidence_threshold,
        }

        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")