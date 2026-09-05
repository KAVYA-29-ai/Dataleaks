import pandas as pd

from dataleaks.detectors.temporal.future_features import FutureFeatureDetector
from dataleaks.detectors.temporal.time_order import TimeOrderDetector
from dataleaks.schemas.dataset import DatasetContext


def make_context(data):
    return DatasetContext(data=data)


def make_split_context(train, test):
    data = pd.concat([train, test], ignore_index=True)

    return DatasetContext(
        data=data,
        train=train,
        test=test,
    )


def test_future_feature_handles_mixed_timezones():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01T10:00:00+05:30",
                "2025-01-02T10:00:00+05:30",
            ],
            "event_time": [
                "2025-01-01T05:00:00+00:00",
                "2025-01-02T06:00:00+00:00",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(data))

    assert len(findings) == 1
    assert findings[0].detector == "temporal_future_features"
    assert findings[0].evidence["future_count"] == 2
    assert findings[0].evidence["comparable_count"] == 2


def test_future_feature_accepts_equivalent_timezones_as_equal():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01T10:00:00+05:30",
            ],
            "event_time": [
                "2025-01-01T04:30:00+00:00",
            ],
        }
    )

    detector = FutureFeatureDetector(
        prediction_time_column="prediction_time",
        feature_time_columns=["event_time"],
    )

    findings = detector.detect(make_context(data))

    assert findings == []


def test_time_order_handles_mixed_timezones():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01T10:00:00+05:30",
                "2025-01-02T10:00:00+05:30",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-01-03T05:00:00+00:00",
                "2025-01-04T05:00:00+00:00",
            ],
        }
    )

    context = make_split_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert findings == []


def test_time_order_detects_violation_across_timezones():
    train = pd.DataFrame(
        {
            "date": [
                "2025-01-01T10:00:00+05:30",
                "2025-01-05T10:00:00+05:30",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "date": [
                "2025-01-03T05:00:00+00:00",
                "2025-01-04T05:00:00+00:00",
            ],
        }
    )

    context = make_split_context(train, test)

    findings = TimeOrderDetector(
        time_column="date",
    ).detect(context)

    assert len(findings) == 1
    assert findings[0].detector == "temporal_time_order"
    assert findings[0].evidence["train_max"] == (
        "2025-01-05 04:30:00+00:00"
    )
    assert findings[0].evidence["test_min"] == (
        "2025-01-03 05:00:00+00:00"
    )