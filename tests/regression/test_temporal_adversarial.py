from __future__ import annotations

import pandas as pd

from dataleaks.detectors.temporal.future_features import (
    FutureFeatureDetector,
)
from dataleaks.detectors.temporal.time_order import (
    TimeOrderDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(
    data: pd.DataFrame,
) -> DatasetContext:
    return DatasetContext(
        data=data,
        target=None,
    )


def test_future_feature_preserves_violating_row_indices():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01T10:00:00+00:00",
                "2025-01-02T10:00:00+00:00",
                "2025-01-03T10:00:00+00:00",
            ],
            "event_time": [
                "2025-01-01T09:00:00+00:00",
                "2025-01-02T11:00:00+00:00",
                "2025-01-03T12:00:00+00:00",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1

    evidence = findings[0].evidence

    assert evidence["future_count"] == 2
    assert evidence["comparable_count"] == 3

    assert evidence["violating_rows"] == [
        {
            "row_index": 1,
            "prediction_time": (
                "2025-01-02 10:00:00+00:00"
            ),
            "feature_time": (
                "2025-01-02 11:00:00+00:00"
            ),
        },
        {
            "row_index": 2,
            "prediction_time": (
                "2025-01-03 10:00:00+00:00"
            ),
            "feature_time": (
                "2025-01-03 12:00:00+00:00"
            ),
        },
    ]


def test_future_feature_evidence_is_bounded():
    rows = 100

    dates = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    data = pd.DataFrame(
        {
            "prediction_time": dates.strftime(
                "%Y-%m-%dT10:00:00+00:00"
            ),
            "event_time": dates.strftime(
                "%Y-%m-%dT11:00:00+00:00"
            ),
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1

    evidence = findings[0].evidence

    assert evidence["future_count"] == 100
    assert evidence["comparable_count"] == 100
    assert evidence["future_ratio"] == 1.0

    assert evidence["evidence_limit"] == 20
    assert len(evidence["violating_rows"]) == 20

    assert evidence["violating_rows"][0] == {
        "row_index": 0,
        "prediction_time": (
            "2025-01-01 10:00:00+00:00"
        ),
        "feature_time": (
            "2025-01-01 11:00:00+00:00"
        ),
    }

    assert evidence["violating_rows"][-1] == {
        "row_index": 19,
        "prediction_time": (
            "2025-01-20 10:00:00+00:00"
        ),
        "feature_time": (
            "2025-01-20 11:00:00+00:00"
        ),
    }


def test_future_feature_equal_timestamp_is_not_future():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01T10:00:00+00:00",
                "2025-01-02T10:00:00+00:00",
            ],
            "event_time": [
                "2025-01-01T10:00:00+00:00",
                "2025-01-02T09:00:00+00:00",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(data))

    assert findings == []


def test_future_feature_invalid_timestamps_are_reported():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01T10:00:00+00:00",
                "not-a-date",
                "2025-01-03T10:00:00+00:00",
            ],
            "event_time": [
                "2025-01-01T09:00:00+00:00",
                "2025-01-02T11:00:00+00:00",
                "not-a-date",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1

    finding = findings[0]
    evidence = finding.evidence

    assert evidence["type"] == "invalid_timestamp"

    assert evidence["invalid_count"] == 2
    assert evidence["invalid_prediction_count"] == 1
    assert evidence["invalid_feature_count"] == 1
    assert evidence["total_rows"] == 3

    assert evidence["invalid_ratio"] == 2 / 3

    assert len(evidence["invalid_rows"]) == 2

    assert evidence["invalid_rows"][0] == {
        "row_index": 1,
        "prediction_time": "not-a-date",
        "feature_time": (
            "2025-01-02T11:00:00+00:00"
        ),
        "prediction_time_invalid": True,
        "feature_time_invalid": False,
    }

    assert evidence["invalid_rows"][1] == {
        "row_index": 2,
        "prediction_time": (
            "2025-01-03T10:00:00+00:00"
        ),
        "feature_time": "not-a-date",
        "prediction_time_invalid": False,
        "feature_time_invalid": True,
    }


def test_time_order_evidence_contains_ordering_details():
    train = pd.DataFrame(
        {
            "event_time": [
                "2025-01-01T10:00:00+00:00",
                "2025-01-05T10:00:00+00:00",
            ]
        }
    )

    test = pd.DataFrame(
        {
            "event_time": [
                "2025-01-03T10:00:00+00:00",
                "2025-01-06T10:00:00+00:00",
            ]
        }
    )

    context = DatasetContext(
        data=pd.concat(
            [train, test],
            ignore_index=True,
        ),
        target=None,
        train=train,
        test=test,
    )

    detector = TimeOrderDetector(
        time_column="event_time",
    )

    findings = detector.detect(context)

    assert len(findings) == 1

    evidence = findings[0].evidence

    assert evidence["type"] == (
        "chronological_split_violation"
    )

    assert evidence["time_column"] == "event_time"

    assert evidence["train_max"] == (
        "2025-01-05 10:00:00+00:00"
    )

    assert evidence["test_min"] == (
        "2025-01-03 10:00:00+00:00"
    )

    assert evidence["overlap_duration"] == (
        "2 days 00:00:00"
    )