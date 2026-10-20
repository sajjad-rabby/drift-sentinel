"""Tests for the ``drift-sentinel`` command-line interface."""

from __future__ import annotations

import json

import pytest

from drift_sentinel.cli import main


@pytest.fixture
def csv_pair(tmp_path, normal_reference, shifted_current):
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    ref.write_text(
        "age,region\n"
        + "".join(f"{v:.4f},emea\n" for v in normal_reference),
        encoding="utf-8",
    )
    cur.write_text(
        "age,region\n"
        + "".join(f"{v:.4f},apac\n" for v in shifted_current),
        encoding="utf-8",
    )
    return str(ref), str(cur)


@pytest.fixture
def llm_log(tmp_path, llm_entries):
    path = tmp_path / "log.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in llm_entries) + "\n", encoding="utf-8"
    )
    return str(path)


class TestBasics:
    def test_no_args_prints_help(self, capsys):
        assert main([]) == 0
        assert "usage:" in capsys.readouterr().out


class TestCompare:
    def test_markdown_output(self, csv_pair, capsys):
        ref, cur = csv_pair
        assert main(["compare", "--reference", ref, "--current", cur]) == 0
        out = capsys.readouterr().out
        assert "## Feature drift" in out
        assert "age" in out

    def test_json_output(self, csv_pair, capsys):
        ref, cur = csv_pair
        assert main(["compare", "--reference", ref, "--current", cur, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "metrics" in payload
        assert payload["metrics"]["verdict"]

    def test_writes_to_file(self, csv_pair, tmp_path):
        ref, cur = csv_pair
        out = tmp_path / "report.md"
        assert main(["compare", "--reference", ref, "--current", cur,
                     "--output", str(out)]) == 0
        assert out.read_text(encoding="utf-8").startswith("# ")

    def test_missing_file_errors(self, tmp_path, capsys):
        assert main(["compare", "--reference", str(tmp_path / "no.csv"),
                     "--current", str(tmp_path / "no.csv")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_fail_on_moderate_triggers_exit_two(self, csv_pair):
        ref, cur = csv_pair
        assert main(["compare", "--reference", ref, "--current", cur,
                     "--json", "--output", "-",
                     "--fail-on", "moderate_drift"]) == 2

    def test_fail_on_significant_not_triggered_by_moderate(self, tmp_path):
        ref = tmp_path / "a.csv"
        cur = tmp_path / "b.csv"
        body = "".join(f"{i}.0,x\n" for i in range(50))
        ref.write_text("n,c\n" + body, encoding="utf-8")
        cur.write_text("n,c\n" + body, encoding="utf-8")
        assert main(["compare", "--reference", str(ref), "--current", str(cur),
                     "--json", "--output", "-",
                     "--fail-on", "significant_drift"]) == 0

    def test_significant_drift_trips_significant_gate(self, csv_pair):
        """The bundled fixture is deliberately shifted, so this must fail the gate."""
        ref, cur = csv_pair
        assert main(["compare", "--reference", ref, "--current", cur,
                     "--json", "--output", "-",
                     "--fail-on", "significant_drift"]) == 2

    def test_alerts_fire_and_report(self, tmp_path, capsys):
        ref = tmp_path / "a.csv"
        cur = tmp_path / "b.csv"
        ref.write_text("n\n" + "".join(f"{i}.0\n" for i in range(50)), encoding="utf-8")
        cur.write_text("n\n" + "".join(f"{i + 100}.0\n" for i in range(50)), encoding="utf-8")
        rules = tmp_path / "rules.json"
        rules.write_text(json.dumps([
            {"metric": "verdict", "operator": "!=", "threshold": "stable"}
        ]), encoding="utf-8")
        assert main(["compare", "--reference", str(ref), "--current", str(cur),
                     "--alerts", str(rules)]) == 0
        out = capsys.readouterr().out
        assert "## Alerts" in out

    def test_invalid_alert_config_errors(self, csv_pair, tmp_path, capsys):
        ref, cur = csv_pair
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert main(["compare", "--reference", ref, "--current", cur,
                     "--alerts", str(bad)]) == 1
        assert "error:" in capsys.readouterr().err

    def test_columns_subcommand(self, csv_pair, capsys):
        ref, _ = csv_pair
        assert main(["columns", "--path", ref]) == 0
        out = capsys.readouterr().out
        assert "numeric" in out and "categorical" in out


class TestLlm:
    def test_markdown_output(self, llm_log, capsys):
        assert main(["llm", "--log", llm_log]) == 0
        out = capsys.readouterr().out
        assert "## LLM metrics" in out
        assert "Refusal rate" in out

    def test_json_output(self, llm_log, capsys):
        assert main(["llm", "--log", llm_log, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["metrics"]["entry_count"] == 40

    def test_reference_must_be_readable(self, llm_log, capsys):
        assert main(["llm", "--log", llm_log, "--reference", "not a file"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_reference_file_works(self, llm_log, tmp_path, capsys):
        ref = tmp_path / "ref.txt"
        ref.write_text("restart the worker", encoding="utf-8")
        assert main(["llm", "--log", llm_log, "--reference", str(ref), "--json"]) == 0
        assert "jaccard_similarity" in json.loads(capsys.readouterr().out)["metrics"]

    def test_malformed_line_errors(self, tmp_path, capsys):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\n{bad}\n', encoding="utf-8")
        assert main(["llm", "--log", str(path)]) == 1
        assert "line 2" in capsys.readouterr().err

    def test_non_object_line_errors(self, tmp_path, capsys):
        path = tmp_path / "arr.jsonl"
        path.write_text('[1,2]\n', encoding="utf-8")
        assert main(["llm", "--log", str(path)]) == 1
        assert "JSON object" in capsys.readouterr().err

    def test_empty_log_errors(self, tmp_path, capsys):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        assert main(["llm", "--log", str(path)]) == 1
        assert "no entries" in capsys.readouterr().err

    def test_price_table_applied(self, llm_log, tmp_path, capsys):
        table = tmp_path / "prices.json"
        table.write_text(json.dumps({
            "test-model": {"prompt_per_1k": 0.003, "completion_per_1k": 0.015}
        }), encoding="utf-8")
        main(["llm", "--log", llm_log, "--price-table", str(table), "--json"])
        assert json.loads(capsys.readouterr().out)["metrics"]["total_cost"] > 0

    def test_fail_on_alert_exits_two(self, tmp_path, capsys):
        entries = [{"response": "I'm sorry, I can't help with that."} for _ in range(10)]
        path = tmp_path / "refuse.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        rules = tmp_path / "rules.json"
        rules.write_text(json.dumps([
            {"metric": "refusal_rate", "operator": ">", "threshold": 0.5}
        ]), encoding="utf-8")
        assert main(["llm", "--log", str(path), "--alerts", str(rules),
                     "--fail-on-alert"]) == 2


class TestReportAndServe:
    def test_report_rerenders_json(self, csv_pair, tmp_path, capsys):
        ref, cur = csv_pair
        report = tmp_path / "r.json"
        main(["compare", "--reference", ref, "--current", cur,
              "--json", "--output", str(report)])
        capsys.readouterr()
        assert main(["report", "--path", str(report)]) == 0
        assert "## Feature drift" in capsys.readouterr().out

    def test_report_missing_file_errors(self, tmp_path, capsys):
        assert main(["report", "--path", str(tmp_path / "no.json")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_serve_reports_bind_failure_instead_of_hanging(self, csv_pair, tmp_path):
        """serve parses the report first, then fails fast on an unusable host.

        A successful bind would block forever serving, so the host is chosen to
        be unbindable; the test asserts the failure is reported, not raised.
        """
        ref, cur = csv_pair
        report = tmp_path / "r.json"
        main(["compare", "--reference", ref, "--current", cur,
              "--json", "--output", str(report)])
        assert main(["serve", "--path", str(report),
                     "--host", "203.0.113.1", "--port", "9"]) == 1

    def test_serve_missing_file_errors(self, tmp_path, capsys):
        assert main(["serve", "--path", str(tmp_path / "no.json")]) == 1
        assert "error:" in capsys.readouterr().err
