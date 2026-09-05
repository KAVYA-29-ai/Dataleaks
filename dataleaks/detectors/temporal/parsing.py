from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimestampParseResult:
    values: pd.Series
    invalid_mask: pd.Series
    ambiguous_mask: pd.Series
    inferred_format: str | None


def _is_date_only_string(value: object) -> bool:
    if not isinstance(value, str):
        return False

    value = value.strip()

    return (
        len(value) == 10
        and value[2] in {"-", "/"}
        and value[5] == value[2]
        and value[:2].isdigit()
        and value[3:5].isdigit()
        and value[6:].isdigit()
    )


def _is_iso_timestamp_string(value: object) -> bool:
    if not isinstance(value, str):
        return False

    value = value.strip()

    if "T" not in value and "t" not in value:
        return False

    try:
        parsed = pd.to_datetime(
            value,
            errors="coerce",
            utc=True,
        )
    except (TypeError, ValueError):
        return False

    return not pd.isna(parsed)


def _parse_with_format(
    series: pd.Series,
    *,
    dayfirst: bool,
) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="coerce",
        format="mixed",
        dayfirst=dayfirst,
        utc=True,
    )


def _infer_date_only_format(
    series: pd.Series,
) -> tuple[str | None, pd.Series]:
    non_null = series.dropna()

    date_only = non_null[
        non_null.map(_is_date_only_string)
    ]

    if date_only.empty:
        return None, pd.Series(
            False,
            index=series.index,
        )

    day_first_evidence = 0
    month_first_evidence = 0

    for value in date_only:
        text = str(value).strip()

        first = int(text[:2])
        second = int(text[3:5])

        if first > 12 and second <= 12:
            day_first_evidence += 1

        elif second > 12 and first <= 12:
            month_first_evidence += 1

    if day_first_evidence > 0 and month_first_evidence == 0:
        return "dayfirst", pd.Series(
            False,
            index=series.index,
        )

    if month_first_evidence > 0 and day_first_evidence == 0:
        return "monthfirst", pd.Series(
            False,
            index=series.index,
        )

    ambiguous = pd.Series(
        False,
        index=series.index,
    )

    if (
        day_first_evidence == 0
        and month_first_evidence == 0
    ):
        for index, value in date_only.items():
            text = str(value).strip()

            first = int(text[:2])
            second = int(text[3:5])

            if first <= 12 and second <= 12:
                ambiguous.loc[index] = True

        if ambiguous.any():
            return None, ambiguous

    return None, ambiguous


def _parse_mixed_series(
    series: pd.Series,
    *,
    inferred_format: str | None,
    ambiguous_mask: pd.Series,
) -> pd.Series:
    """
    Parse timestamps while applying inferred day/month convention only
    to date-only strings.

    ISO timestamps are parsed independently so that dayfirst/monthfirst
    inference cannot reinterpret an already-unambiguous ISO timestamp.
    """

    values = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns, UTC]",
    )

    non_null = series.notna()

    iso_mask = non_null & series.map(
        _is_iso_timestamp_string
    )

    if iso_mask.any():
        values.loc[iso_mask] = pd.to_datetime(
            series.loc[iso_mask],
            errors="coerce",
            format="mixed",
            utc=True,
        )

    date_only_mask = non_null & series.map(
        _is_date_only_string
    )

    parse_date_only_mask = (
        date_only_mask
        & ~ambiguous_mask
    )

    if parse_date_only_mask.any():
        if inferred_format == "dayfirst":
            values.loc[parse_date_only_mask] = (
                pd.to_datetime(
                    series.loc[parse_date_only_mask],
                    errors="coerce",
                    format="mixed",
                    dayfirst=True,
                    utc=True,
                )
            )

        elif inferred_format == "monthfirst":
            values.loc[parse_date_only_mask] = (
                pd.to_datetime(
                    series.loc[parse_date_only_mask],
                    errors="coerce",
                    format="mixed",
                    dayfirst=False,
                    utc=True,
                )
            )

        else:
            values.loc[parse_date_only_mask] = (
                pd.to_datetime(
                    series.loc[parse_date_only_mask],
                    errors="coerce",
                    format="mixed",
                    utc=True,
                )
            )

    other_mask = (
        non_null
        & ~iso_mask
        & ~date_only_mask
    )

    if other_mask.any():
        values.loc[other_mask] = pd.to_datetime(
            series.loc[other_mask],
            errors="coerce",
            format="mixed",
            utc=True,
        )

    values = values.mask(
        ambiguous_mask,
        pd.NaT,
    )

    return values


def parse_timestamps(
    series: pd.Series,
) -> TimestampParseResult:
    """
    Parse timestamps without silently guessing ambiguous date-only values.

    When unambiguous date-only values establish a column-wide convention,
    that convention is applied consistently to ambiguous date-only values.

    ISO timestamps are parsed independently and are never reinterpreted
    using the inferred date-only convention.
    """

    inferred_format, ambiguous_mask = (
        _infer_date_only_format(series)
    )

    values = _parse_mixed_series(
        series,
        inferred_format=inferred_format,
        ambiguous_mask=ambiguous_mask,
    )

    invalid_mask = (
        series.notna()
        & values.isna()
        & ~ambiguous_mask
    )

    return TimestampParseResult(
        values=values,
        invalid_mask=invalid_mask,
        ambiguous_mask=ambiguous_mask,
        inferred_format=inferred_format,
    )