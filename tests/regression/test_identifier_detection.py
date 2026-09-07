from __future__ import annotations

import pandas as pd

from dataleaks import DataLeaks


def test_camel_case_identifier_is_detected():
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(100)],
            "value": list(range(100)),
            "target": [0, 1] * 50,
        }
    )

    report = DataLeaks(df, target="target").run()

    findings = [
        finding
        for finding in report.findings
        if finding.detector == "feature_identifier"
    ]

    assert any("customerID" in finding.affected_columns for finding in findings)


def test_numeric_measurement_is_not_split_overlap_key():
    train = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(100)],
            "TotalCharges": [100.0 + i for i in range(100)],
            "target": [0, 1] * 50,
        }
    )
    test = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(50, 150)],
            "TotalCharges": [150.0 + i for i in range(100)],
            "target": [0, 1] * 50,
        }
    )

    report = DataLeaks(
        train,
        target="target",
        test=test,
    ).run()

    overlap_findings = [
        finding
        for finding in report.findings
        if finding.detector == "split_overlap"
    ]

    affected = {
        column
        for finding in overlap_findings
        for column in finding.affected_columns
    }

    assert "customerID" in affected
    assert "TotalCharges" not in affected
