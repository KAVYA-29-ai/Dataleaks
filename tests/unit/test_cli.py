import json

import pandas as pd

from dataleaks.cli import main


def test_cli_runs_with_csv_and_target(tmp_path, monkeypatch, capsys):
    data = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "feature": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(csv_path),
            "--target",
            "target",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "DataLeaks Report" in captured.out
    assert "Risk Level:" in captured.out
    assert "Findings:" in captured.out


def test_cli_runs_without_target(tmp_path, monkeypatch, capsys):
    data = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(csv_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "DataLeaks Report" in captured.out


def test_cli_supports_json_output(tmp_path, monkeypatch, capsys):
    data = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )

    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(csv_path),
            "--target",
            "target",
            "--output",
            "json",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 0

    payload = json.loads(captured.out)

    assert "risk" in payload
    assert "summary" in payload
    assert "findings" in payload
    assert "recommendations" in payload


def test_cli_returns_error_for_missing_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    missing_path = tmp_path / "missing.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(missing_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "dataset file not found" in captured.err


def test_cli_returns_error_for_missing_target(
    tmp_path,
    monkeypatch,
    capsys,
):
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(csv_path),
            "--target",
            "does_not_exist",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "does not exist" in captured.err

def test_cli_help_exits_successfully(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            "--help",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()

    assert "Detect data leakage" in captured.out
    assert "--target TARGET" in captured.out


def test_cli_rejects_invalid_output_format(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            "data.csv",
            "--output",
            "xml",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2

    captured = capsys.readouterr()

    assert "invalid choice" in captured.err


def test_cli_handles_empty_csv(
    tmp_path,
    monkeypatch,
    capsys,
):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(csv_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "dataset file is empty" in captured.err


def test_cli_handles_malformed_csv(
    tmp_path,
    monkeypatch,
    capsys,
):
    csv_path = tmp_path / "malformed.csv"

    csv_path.write_text(
        "a,b,c\n"
        "1,2,3\n"
        '"broken,2,3\n'
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "dataleaks",
            str(csv_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "could not parse CSV file" in captured.err