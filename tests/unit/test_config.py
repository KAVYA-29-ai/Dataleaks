import pytest

from dataleaks.schemas.config import DataLeaksConfig


def test_default_config():
    config = DataLeaksConfig()

    assert config.enable_target_checks is True
    assert config.enable_split_checks is True
    assert config.enable_temporal_checks is True
    assert config.confidence_threshold == 0.5


def test_custom_config():
    config = DataLeaksConfig(
        enable_temporal_checks=False,
        confidence_threshold=0.8,
    )

    assert config.enable_temporal_checks is False
    assert config.confidence_threshold == 0.8


@pytest.mark.parametrize(
    "field",
    [
        "duplicate_threshold",
        "near_duplicate_threshold",
        "confidence_threshold",
    ],
)
def test_thresholds_must_be_valid(field):
    with pytest.raises(ValueError):
        DataLeaksConfig(**{field: 1.5})