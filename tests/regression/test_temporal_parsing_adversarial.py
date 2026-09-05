import pandas as pd

from dataleaks.detectors.temporal.parsing import parse_timestamps


def test_day_first_convention_is_applied_to_ambiguous_dates():
    series = pd.Series(
        [
            "02-09-2023",
            "15-08-2023",
            "21-03-2023",
        ]
    )

    result = parse_timestamps(series)

    assert result.inferred_format == "dayfirst"

    assert result.values.iloc[0] == pd.Timestamp(
        "2023-09-02",
        tz="UTC",
    )

    assert result.values.iloc[1] == pd.Timestamp(
        "2023-08-15",
        tz="UTC",
    )

    assert result.values.iloc[2] == pd.Timestamp(
        "2023-03-21",
        tz="UTC",
    )

    assert not result.ambiguous_mask.any()


def test_month_first_convention_is_applied_to_ambiguous_dates():
    series = pd.Series(
        [
            "09-15-2023",
            "03-21-2023",
            "02-09-2023",
        ]
    )

    result = parse_timestamps(series)

    assert result.inferred_format == "monthfirst"

    assert result.values.iloc[0] == pd.Timestamp(
        "2023-09-15",
        tz="UTC",
    )

    assert result.values.iloc[1] == pd.Timestamp(
        "2023-03-21",
        tz="UTC",
    )

    assert result.values.iloc[2] == pd.Timestamp(
        "2023-02-09",
        tz="UTC",
    )


def test_ambiguous_dates_without_convention_are_not_guessed():
    series = pd.Series(
        [
            "02-09-2023",
            "03-04-2023",
            "11-12-2023",
        ]
    )

    result = parse_timestamps(series)

    assert result.inferred_format is None

    assert result.ambiguous_mask.all()

    assert result.values.isna().all()


def test_iso_timestamps_do_not_override_date_only_convention():
    series = pd.Series(
        [
            "2023-09-02T10:00:00+00:00",
            "15-08-2023",
            "02-09-2023",
        ]
    )

    result = parse_timestamps(series)

    assert result.inferred_format == "dayfirst"

    assert result.values.iloc[0] == pd.Timestamp(
        "2023-09-02T10:00:00+00:00"
    )

    assert result.values.iloc[1] == pd.Timestamp(
        "2023-08-15",
        tz="UTC",
    )

    assert result.values.iloc[2] == pd.Timestamp(
        "2023-09-02",
        tz="UTC",
    )