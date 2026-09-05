import pandas as pd
import pytest

from dataleaks import DataLeaks


def test_known_exact_target_leakage():
    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35],
            "target": [0, 1, 0, 1],
            "target_copy": [0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "target_direct"
    ]

    assert len(findings) == 1

    finding = findings[0]

    assert finding.severity == "critical"
    assert finding.confidence == 1.0
    assert "target_copy" in finding.affected_columns


def test_known_derived_target_leakage():
    data = pd.DataFrame(
        {
            "target": [1, 2, 3, 4],
            "derived": [11, 14, 17, 20],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "target_derived"
    ]

    assert len(findings) == 1
    assert findings[0].severity == "critical"


def test_known_identifier_feature():
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "age": [20, 25, 30, 35],
            "target": [0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "feature_identifier"
    ]

    assert len(findings) == 1
    assert findings[0].affected_columns == [
        "customer_id"
    ]


def test_known_train_test_duplicate():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    train = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [2, 4],
            "target": [1, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        train=train,
        test=test,
    ).run()

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "split_duplicates"
    ]

    assert len(findings) == 1
    assert findings[0].confidence > 0.0


def test_known_temporal_future_feature():
    data = pd.DataFrame(
        {
            "prediction_time": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ],
            "feature_time": [
                "2025-01-02",
                "2025-01-01",
                "2025-01-04",
            ],
            "target": [0, 1, 0],
        }
    )

    report = DataLeaks(
        data,
        target="target",
        metadata={
            "temporal": {
                "prediction_time_column": "prediction_time",
                "feature_time_columns": [
                    "feature_time",
                ],
            }
        },
    ).run()

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "temporal_future_features"
    ]

    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].confidence == pytest.approx(2 / 3)


def test_clean_dataset_remains_clean():
    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "income": [
                30000,
                40000,
                50000,
                60000,
                70000,
                80000,
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    report = DataLeaks(
        data,
        target="target",
    ).run()

    assert report.findings == []
    assert report.risk_score == 0.0
    assert report.risk_level == "low"