import pytest

from dataleaks.scoring.severity import Severity


def test_severity_levels_are_defined():
    assert Severity.LEVELS == {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }


def test_normalize_lowercases_severity():
    assert Severity.normalize("HIGH") == "high"


def test_normalize_strips_whitespace():
    assert Severity.normalize("  critical  ") == "critical"


def test_score_returns_numeric_severity():
    assert Severity.score("low") == 1
    assert Severity.score("medium") == 2
    assert Severity.score("high") == 3
    assert Severity.score("critical") == 4


def test_is_at_least_returns_true_for_equal_level():
    assert Severity.is_at_least(
        "high",
        "high",
    )


def test_is_at_least_returns_true_for_higher_level():
    assert Severity.is_at_least(
        "critical",
        "high",
    )


def test_is_at_least_returns_false_for_lower_level():
    assert not Severity.is_at_least(
        "medium",
        "high",
    )


def test_highest_returns_highest_severity():
    assert Severity.highest(
        "low",
        "critical",
        "medium",
    ) == "critical"


def test_highest_works_case_insensitively():
    assert Severity.highest(
        "low",
        "HIGH",
        "medium",
    ) == "high"


def test_highest_rejects_empty_input():
    with pytest.raises(ValueError):
        Severity.highest()


def test_invalid_severity_raises_value_error():
    with pytest.raises(ValueError):
        Severity.normalize("unknown")


def test_non_string_severity_raises_type_error():
    with pytest.raises(TypeError):
        Severity.normalize(123)