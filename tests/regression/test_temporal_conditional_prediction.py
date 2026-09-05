import json
import sys

from dataleaks.cli import main


def test_conditional_prediction_timestamp_missingness_is_ignored(
    monkeypatch,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "temporal.csv"

    csv_path.write_text(
        "customer_id,churn,outcome_date,last_activity_date\n"
        "1,1,02-09-2023,not-a-date\n"
        "2,0,,15-08-2023\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataleaks",
            str(csv_path),
            "--target",
            "churn",
            "--prediction-time-column",
            "outcome_date",
            "--feature-time-columns",
            "last_activity_date",
            "--conditional-time-column",
            "outcome_date",
            "--conditional-target-column",
            "churn",
            "--conditional-present-when",
            "1",
            "--output",
            "json",
        ],
    )

    exit_code = main()

    assert exit_code == 0

    output = json.loads(capsys.readouterr().out)

    findings = output["findings"]

    invalid_findings = [
        finding
        for finding in findings
        if finding["evidence"].get("type") == "invalid_timestamp"
    ]

    assert invalid_findings

    evidence = invalid_findings[0]["evidence"]

    assert evidence["conditional_configured"] is True
    assert evidence["invalid_prediction_count"] == 0
    assert evidence["invalid_feature_count"] == 1