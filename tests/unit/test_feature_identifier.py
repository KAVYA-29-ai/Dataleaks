import pandas as pd
import pytest

from dataleaks.detectors.feature.identifier import (
    IdentifierLeakageDetector,
)
from dataleaks.schemas.dataset import DatasetContext


def make_context(
    data,
    train=None,
    test=None,
    target=None,
    metadata=None,
):
    return DatasetContext(
        data=data,
        train=train,
        test=test,
        target=target,
        metadata=metadata or {},
    )


def test_detects_identifier_like_column():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "age": [20, 25, 30, 35],
            "target": [0, 1, 0, 1],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(
            df,
            target="target",
        )
    )

    assert len(findings) == 1
    assert findings[0].detector == "feature_identifier"
    assert findings[0].category == "feature_leakage"
    assert findings[0].affected_columns == ["user_id"]


def test_high_cardinality_column_without_identifier_signal_is_clean():
    df = pd.DataFrame(
        {
            "feature": [
                "a",
                "b",
                "c",
                "d",
                "e",
            ]
        }
    )

    findings = IdentifierLeakageDetector(
        uniqueness_threshold=0.9
    ).detect(
        make_context(df)
    )

    assert findings == []


def test_regular_low_cardinality_feature_is_clean():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Delhi",
                "Mumbai",
                "Delhi",
                "Mumbai",
            ]
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(df)
    )

    assert findings == []


def test_target_column_is_ignored():
    df = pd.DataFrame(
        {
            "target_id": [1, 2, 3, 4],
            "target": [1, 2, 3, 4],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(
            df,
            target="target_id",
        )
    )

    assert findings == []


def test_train_test_identifier_overlap_is_detected():
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [3, 4, 5],
            "value": [40, 50, 60],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(
            data,
            train=train,
            test=test,
        )
    )

    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert (
        findings[0].evidence["train_test_overlap_count"]
        == 1
    )


def test_non_overlapping_identifier_has_medium_severity():
    train = pd.DataFrame(
        {
            "user_id": [1, 2],
        }
    )

    test = pd.DataFrame(
        {
            "user_id": [3, 4],
        }
    )

    data = pd.concat(
        [train, test],
        ignore_index=True,
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(
            data,
            train=train,
            test=test,
        )
    )

    assert len(findings) == 1
    assert findings[0].severity == "medium"


def test_missing_values_are_excluded_from_uniqueness():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, None, 3, None],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert (
        findings[0].evidence["uniqueness_ratio"]
        == 1.0
    )


def test_custom_uniqueness_threshold_does_not_override_identifier_signal():
    df = pd.DataFrame(
        {
            "feature": [
                "a",
                "b",
                "b",
                "c",
            ],
        }
    )

    findings = IdentifierLeakageDetector(
        uniqueness_threshold=0.75
    ).detect(
        make_context(df)
    )

    assert findings == []


def test_custom_uniqueness_threshold_affects_identifier_severity():
    df = pd.DataFrame(
        {
            "user_id": [
                "a",
                "b",
                "b",
                "c",
            ],
        }
    )

    findings = IdentifierLeakageDetector(
        uniqueness_threshold=0.75
    ).detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].severity == "medium"
    assert findings[0].confidence == 0.75


def test_empty_dataframe_produces_no_findings():
    df = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="object"),
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(df)
    )

    assert findings == []


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        IdentifierLeakageDetector(
            uniqueness_threshold=1.5
        )


def test_identifier_pattern_is_case_insensitive():
    df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 2, 3, 4],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(df)
    )

    assert len(findings) == 1


def test_explicit_identifier_metadata_is_supported():
    df = pd.DataFrame(
        {
            "customer_key": [
                "a",
                "b",
                "c",
                "d",
            ],
        }
    )

    findings = IdentifierLeakageDetector().detect(
        make_context(
            df,
            metadata={
                "identifier_columns": [
                    "customer_key",
                ]
            },
        )
    )

    assert len(findings) == 1
    assert findings[0].evidence["explicit_identifier"] is True


def test_invalid_identifier_metadata_raises_error():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
        }
    )

    with pytest.raises(TypeError):
        IdentifierLeakageDetector().detect(
            make_context(
                df,
                metadata={
                    "identifier_columns": "user_id",
                },
            )
        )
def test_low_severity_identifier_confidence_tracks_uniqueness():
    df = pd.DataFrame(
        {
            "leaky_id": [
                "a",
                "b",
                "c",
                "d",
                "d",
                "e",
                "e",
                "f",
                "g",
                "h",
            ],
        }
    )

    findings = IdentifierLeakageDetector(
        uniqueness_threshold=0.95
    ).detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].severity == "low"
    assert findings[0].confidence == pytest.approx(0.8)
    assert (
        findings[0].evidence["uniqueness_ratio"]
        == pytest.approx(0.8)
    )


def test_confidence_is_continuous_across_uniqueness_threshold():
    df = pd.DataFrame(
        {
            "user_id": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "h",
                "i",
            ],
        }
    )

    findings = IdentifierLeakageDetector(
        uniqueness_threshold=0.9
    ).detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].severity == "medium"
    assert findings[0].confidence == pytest.approx(0.9)


def test_confidence_below_threshold_is_not_forced_to_half():
    df = pd.DataFrame(
        {
            "user_id": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "h",
                "h",
            ],
        }
    )

    findings = IdentifierLeakageDetector(
        uniqueness_threshold=0.95
    ).detect(
        make_context(df)
    )

    assert len(findings) == 1
    assert findings[0].severity == "low"
    assert findings[0].confidence == pytest.approx(0.8)
    assert findings[0].confidence != 0.5