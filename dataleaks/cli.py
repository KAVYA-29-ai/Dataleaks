from __future__ import annotations

import argparse
import ast
import sys

import pandas as pd

from dataleaks.api import DataLeaks
from dataleaks.reporting.console import ConsoleReporter
from dataleaks.reporting.json import JSONReporter


def build_parser() -> argparse.ArgumentParser:
    """Build the DataLeaks command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="dataleaks",
        description="Detect data leakage in machine learning datasets.",
    )

    parser.add_argument(
        "file",
        help="Path to the training CSV dataset.",
    )

    parser.add_argument(
        "--target",
        help="Target column used for leakage analysis.",
        default=None,
    )

    parser.add_argument(
        "--test",
        help="Path to the test CSV dataset.",
        default=None,
    )

    parser.add_argument(
        "--validation",
        help="Path to the validation CSV dataset.",
        default=None,
    )

    parser.add_argument(
        "--time-column",
        help=(
            "Time column used for chronological train/test "
            "split validation."
        ),
        default=None,
    )

    parser.add_argument(
        "--prediction-time-column",
        help=(
            "Prediction timestamp column used to detect "
            "future feature timestamps."
        ),
        default=None,
    )

    parser.add_argument(
        "--feature-time-columns",
        nargs="+",
        help=(
            "One or more feature timestamp columns to compare "
            "against the prediction timestamp."
        ),
        default=None,
    )

    parser.add_argument(
        "--conditional-time-column",
        help=(
            "Temporal column whose expected presence depends "
            "on a target condition."
        ),
        default=None,
    )

    parser.add_argument(
        "--conditional-target-column",
        help=(
            "Target column controlling conditional temporal "
            "missingness."
        ),
        default=None,
    )

    parser.add_argument(
        "--conditional-present-when",
        help=(
            "Target value for which the conditional temporal "
            "column is expected to be present."
        ),
        default=None,
    )

    parser.add_argument(
        "--output",
        choices=["console", "json"],
        default="console",
        help="Output format. Defaults to console.",
    )

    return parser


def _parse_cli_value(value: str) -> object:
    """Parse a CLI scalar into a Python value when possible."""

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _build_temporal_metadata(
    *,
    time_column: str | None,
    prediction_time_column: str | None,
    feature_time_columns: list[str] | None,
    conditional_time_column: str | None = None,
    conditional_target_column: str | None = None,
    conditional_present_when: str | None = None,
) -> dict:
    """Build temporal metadata from CLI arguments."""

    temporal: dict[str, object] = {}

    if time_column is not None:
        temporal["time_column"] = time_column

    if prediction_time_column is not None:
        temporal["prediction_time_column"] = (
            prediction_time_column
        )

    if feature_time_columns is not None:
        temporal["feature_time_columns"] = (
            feature_time_columns
        )

    conditional_args = (
        conditional_time_column,
        conditional_target_column,
        conditional_present_when,
    )

    conditional_count = sum(
        value is not None
        for value in conditional_args
    )

    if conditional_count not in (0, 3):
        raise ValueError(
            "--conditional-time-column, "
            "--conditional-target-column, and "
            "--conditional-present-when must be provided together"
        )

    if conditional_count == 3:
        temporal["conditional_columns"] = {
            conditional_time_column: {
                "target_column": conditional_target_column,
                "present_when": _parse_cli_value(
                    conditional_present_when
                ),
            }
        }

    return temporal


def main() -> int:
    """Run the DataLeaks CLI."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.file)

        test = (
            pd.read_csv(args.test)
            if args.test is not None
            else None
        )

        validation = (
            pd.read_csv(args.validation)
            if args.validation is not None
            else None
        )

        temporal_metadata = _build_temporal_metadata(
            time_column=args.time_column,
            prediction_time_column=(
                args.prediction_time_column
            ),
            feature_time_columns=args.feature_time_columns,
            conditional_time_column=(
                args.conditional_time_column
            ),
            conditional_target_column=(
                args.conditional_target_column
            ),
            conditional_present_when=(
                args.conditional_present_when
            ),
        )

        metadata = {}

        if temporal_metadata:
            metadata["temporal"] = temporal_metadata

        report = DataLeaks(
            data,
            target=args.target,
            test=test,
            validation=validation,
            metadata=metadata,
        ).run()

        if args.output == "json":
            output = JSONReporter().render(report)
        else:
            output = ConsoleReporter().render(report)

        print(output)
        return 0

    except FileNotFoundError as exc:
        missing_file = args.file

        if args.test is not None and args.test in str(exc):
            missing_file = args.test

        elif (
            args.validation is not None
            and args.validation in str(exc)
        ):
            missing_file = args.validation

        print(
            f"Error: dataset file not found: {missing_file}",
            file=sys.stderr,
        )
        return 1

    except pd.errors.EmptyDataError as exc:
        print(
            f"Error: dataset file is empty: {exc}",
            file=sys.stderr,
        )
        return 1

    except pd.errors.ParserError as exc:
        print(
            f"Error: could not parse CSV file: {exc}",
            file=sys.stderr,
        )
        return 1

    except (TypeError, ValueError) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())