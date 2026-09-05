import json
import sys

from dataleaks.cli import main


def test_cli_wires_conditional_temporal_configuration(
    monkeypatch,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "temporal.csv"

    csv_path.write_text(
        "customer_id,churn,outcome_date,last_activity_date\n"
        "1,1,02-09-2023,01-09-2023\n"
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

    temporal = output["metadata"]["temporal"]
    conditional = temporal["conditional_columns"]["outcome_date"]

    assert conditional["target_column"] == "churn"
    assert conditional["present_when"] == 1