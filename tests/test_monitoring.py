"""Tests for drift comparison, LLM monitoring, alerts, CSV I/O, reports, and export."""

from __future__ import annotations

import json
import math

import pytest

from drift_sentinel.alerts import (
    Alert,
    AlertRule,
    ConsoleAlerter,
    WebhookAlerter,
    build_alerters,
    evaluate_alert_rules,
    evaluate_operator,
    get_operator,
    lookup_metric,
)
from drift_sentinel.csv_io import read_csv, split_numeric_categorical, write_csv
from drift_sentinel.drift import compare_datasets, compare_feature
from drift_sentinel.llm_monitors import (
    LLMMetrics,
    compute_llm_metrics,
    estimate_cost,
    is_refusal,
    jaccard_similarity,
)
from drift_sentinel.prometheus import render_metrics
from drift_sentinel.report import build_json_report, build_markdown_report


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------

class TestCompareFeature:
    def test_numeric_features_use_psi(self):
        result = compare_feature("x", [float(i) for i in range(100)],
                                 [float(i) + 50 for i in range(100)])
        assert result.kind == "numeric"
        assert result.metric == "psi"
        assert result.value > 0

    def test_categorical_features_use_js(self):
        result = compare_feature("x", ["a"] * 50 + ["b"] * 50,
                                 ["a"] * 5 + ["b"] * 95)
        assert result.kind == "categorical"
        assert result.metric == "js_distance"

    def test_empty_reference_returns_none(self):
        assert compare_feature("x", [], [1.0, 2.0]) is None

    def test_empty_current_returns_none(self):
        assert compare_feature("x", ["a", "b"], []) is None

    def test_single_row_numeric_is_skipped(self):
        """Regression: PSI needs a reference distribution of at least 2."""
        assert compare_feature("x", [1.0], [2.0]) is None

    def test_numeric_with_two_rows_works(self):
        assert compare_feature("x", [1.0, 2.0], [10.0, 20.0]) is not None

    def test_mixed_types_fall_back_to_categorical(self):
        """A numeric column that picks up text in production must not crash."""
        result = compare_feature("x", [1.0, 2.0, 3.0, 4.0], ["a", "b", "c", "d"])
        assert result.kind == "categorical"

    def test_custom_psi_bands_change_classification(self):
        ref = [float(i) for i in range(200)]
        cur = [float(i) + 20 for i in range(200)]
        strict = compare_feature("x", ref, cur, psi_bands={"stable": 0.001, "moderate": 0.002})
        lenient = compare_feature("x", ref, cur, psi_bands={"stable": 10.0, "moderate": 20.0})
        assert strict.band == "significant"
        assert lenient.band == "stable"

    def test_blank_cells_are_ignored(self):
        assert compare_feature("x", ["1", "", "3"], ["1", "", "3"]) is not None

    def test_to_dict_rounds_values(self):
        result = compare_feature("x", [float(i) for i in range(100)],
                                 [float(i) + 10 for i in range(100)])
        payload = result.to_dict()
        assert payload["feature"] == "x"
        assert isinstance(payload["value"], float)


class TestCompareDatasets:
    @pytest.fixture
    def data(self, normal_reference, shifted_current):
        return (
            {"age": normal_reference, "region": ["emea"] * 300 + ["amer"] * 300},
            {"age": shifted_current, "region": ["apac"] * 600},
        )

    def test_only_shared_columns_compared(self, data):
        ref = {"a": [1.0, 2.0, 3.0], "only_ref": [1.0, 2.0, 3.0]}
        cur = {"a": [1.0, 2.0, 4.0], "only_cur": [1.0, 2.0, 3.0]}
        report = compare_datasets(ref, cur)
        assert [f.feature for f in report.features] == ["a"]

    def test_verdict_is_significant_when_any_feature_is(self, normal_reference, shifted_current):
        report = compare_datasets({"age": normal_reference}, {"age": shifted_current})
        assert report.verdict == "significant_drift"

    def test_verdict_is_stable_for_identical(self, normal_reference):
        report = compare_datasets({"age": normal_reference}, {"age": list(normal_reference)})
        assert report.verdict == "stable"

    def test_no_shared_columns_is_no_data(self):
        assert compare_datasets({"a": [1.0]}, {"b": [1.0]}).verdict == "no_data"

    def test_features_are_sorted_by_name(self, data):
        report = compare_datasets(*data)
        names = [f.feature for f in report.features]
        assert names == sorted(names)

    def test_report_to_dict_shape(self, normal_reference, shifted_current):
        payload = compare_datasets({"age": normal_reference}, {"age": shifted_current}).to_dict()
        assert set(payload) == {"verdict", "features"}
        assert isinstance(payload["features"], list)

    def test_all_features_skipped_is_no_data(self):
        assert compare_datasets({"a": []}, {"a": []}).verdict == "no_data"


# --------------------------------------------------------------------------
# llm monitors
# --------------------------------------------------------------------------

class TestRefusalDetection:
    @pytest.mark.parametrize("text", [
        "I'm sorry, I can't help with that.",
        "I cannot assist with that request.",
        "I can't do that.",
        "I'm unable to provide that.",
        "As an AI language model, I won't.",
    ])
    def test_detects_refusals(self, text):
        assert is_refusal(text) is True

    def test_unmatched_apology_is_not_a_refusal(self):
        """Phrase matching is exact: a mere apology is not a refusal."""
        assert is_refusal("Sorry, that is not something I can help with.") is False

    @pytest.mark.parametrize("text", [
        "Restart the ingestion worker.",
        "Check the dashboard for the lag metric.",
        "",
    ])
    def test_accepts_normal_responses(self, text):
        assert is_refusal(text) is False


class TestJaccard:
    def test_identical_text_is_one(self):
        assert jaccard_similarity(["the quick brown fox"], "the quick brown fox") == 1.0

    def test_disjoint_text_is_zero(self):
        assert jaccard_similarity(["alpha"], "beta") == 0.0

    def test_partial_overlap(self):
        value = jaccard_similarity(["a b c"], "a b d")
        assert 0.0 < value < 1.0

    def test_empty_responses_do_not_crash(self):
        assert jaccard_similarity(["", ""], "reference") == 0.0

    def test_is_bounded(self, llm_entries):
        value = jaccard_similarity([e["response"] for e in llm_entries], "runbook")
        assert 0.0 <= value <= 1.0


class TestCostEstimation:
    def test_known_model(self, price_table):
        cost = estimate_cost(1000, 1000, "test-model", price_table)
        assert cost == pytest.approx(0.003 + 0.015)

    def test_unknown_model_is_free(self, price_table):
        assert estimate_cost(1000, 1000, "unknown", price_table) == 0.0

    def test_zero_tokens_is_free(self, price_table):
        assert estimate_cost(0, 0, "test-model", price_table) == 0.0


class TestLlmMetrics:
    def test_computes_expected_keys(self, llm_entries):
        metrics = compute_llm_metrics(llm_entries).to_dict()
        for key in ("entry_count", "total_prompt_tokens", "total_cost", "latency_ms"):
            assert key in metrics

    def test_entry_count(self, llm_entries):
        assert compute_llm_metrics(llm_entries).to_dict()["entry_count"] == len(llm_entries)

    def test_refusal_rate_is_a_fraction(self):
        entries = [{"response": "ok"}, {"response": "I'm sorry, I can't help."}]
        rate = compute_llm_metrics(entries).to_dict()["refusal_rate"]
        assert 0.0 <= rate <= 1.0
        assert rate == pytest.approx(0.5)

    def test_latency_percentiles_present(self, llm_entries):
        latency = compute_llm_metrics(llm_entries).to_dict()["latency_ms"]
        for key in ("count", "mean", "p50", "p90", "p99", "max", "min"):
            assert key in latency

    def test_jaccard_only_when_reference_given(self, llm_entries):
        without = compute_llm_metrics(llm_entries).to_dict()
        with_ref = compute_llm_metrics(llm_entries, reference_response="worker").to_dict()
        assert "jaccard_similarity" not in without
        assert "jaccard_similarity" in with_ref

    def test_per_model_breakdown(self, llm_entries):
        metrics = compute_llm_metrics(llm_entries).to_dict()
        assert "test-model" in metrics["per_model"]

    def test_cost_uses_price_table(self, llm_entries, price_table):
        metrics = compute_llm_metrics(llm_entries, price_table=price_table).to_dict()
        assert metrics["total_cost"] > 0

    def test_no_price_table_means_zero_cost(self, llm_entries):
        assert compute_llm_metrics(llm_entries).to_dict()["total_cost"] == 0.0

    def test_entries_without_latency_are_skipped(self):
        metrics = compute_llm_metrics([{"response": "hi"}]).to_dict()
        assert metrics["entry_count"] == 1

    def test_empty_input(self):
        assert compute_llm_metrics([]).to_dict()["entry_count"] == 0


# --------------------------------------------------------------------------
# alerts
# --------------------------------------------------------------------------

class TestOperators:
    @pytest.mark.parametrize("op,expected", [
        (">", False), (">=", False), ("<", True), ("<=", True), ("==", False), ("!=", True)
    ])
    def test_numeric_comparisons(self, op, expected):
        assert evaluate_operator(5, op, 10) is expected

    def test_equal_values(self):
        assert evaluate_operator(10, "==", 10) is True
        assert evaluate_operator(10, "<=", 10) is True

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError):
            get_operator("~=")

    def test_string_equality(self):
        assert evaluate_operator("abc", "==", "abc") is True

    def test_boolean_thresholds(self):
        assert evaluate_operator(True, "==", True) is True


class TestLookupMetric:
    def test_top_level_key(self):
        assert lookup_metric({"a": 1}, "a") == 1

    def test_dotted_key(self):
        assert lookup_metric({"latency_ms": {"p99": 5}}, "latency_ms.p99") == 5

    def test_deeply_nested(self):
        assert lookup_metric({"a": {"b": {"c": 7}}}, "a.b.c") == 7

    def test_missing_raises_key_error(self):
        """A rule naming a metric that does not exist is an error, not a no-op."""
        with pytest.raises(KeyError):
            lookup_metric({"a": 1}, "b")

    def test_missing_intermediate_raises_key_error(self):
        with pytest.raises(KeyError):
            lookup_metric({"a": 1}, "a.b.c")

    def test_zero_is_not_missing(self):
        assert lookup_metric({"a": 0}, "a") == 0


class TestAlertRules:
    def test_rule_fires_above_threshold(self):
        rule = AlertRule("refusal_rate", ">", 0.1)
        alert = rule.check({"refusal_rate": 0.5})
        assert alert is not None
        assert alert.actual == 0.5

    def test_rule_silent_below_threshold(self):
        assert AlertRule("refusal_rate", ">", 0.9).check({"refusal_rate": 0.5}) is None

    def test_missing_metric_never_fires(self):
        assert AlertRule("nonexistent", ">", 0).check({"a": 1}) is None

    def test_dotted_metric(self):
        assert AlertRule("latency_ms.p99", ">", 1000).check({"latency_ms": {"p99": 2000}}) is not None

    def test_from_dict(self):
        rule = AlertRule.from_dict({"metric": "m", "operator": "<", "threshold": 1})
        assert rule.metric == "m" and rule.threshold == 1

    def test_from_dict_rejects_bad_operator(self):
        with pytest.raises(ValueError):
            AlertRule.from_dict({"metric": "m", "operator": "~", "threshold": 1})

    def test_alert_formats_to_console(self):
        rule = AlertRule("m", ">", 1, message="watch out")
        text = rule.check({"m": 5}).format_console()
        assert "watch out" in text

    def test_alert_to_dict(self):
        rule = AlertRule("m", ">", 1)
        payload = rule.check({"m": 5}).to_dict()
        assert payload["metric"] == "m" and payload["actual"] == 5


class TestEvaluateAlertRules:
    def test_multiple_rules(self):
        fired = evaluate_alert_rules(
            {"refusal_rate": 0.9, "latency_ms": {"p90": 5000}},
            [
                {"metric": "refusal_rate", "operator": ">", "threshold": 0.5},
                {"metric": "latency_ms.p90", "operator": ">", "threshold": 1000},
                {"metric": "refusal_rate", "operator": ">", "threshold": 0.99},
            ],
        )
        assert len(fired) == 2

    def test_no_rules(self):
        assert evaluate_alert_rules({"a": 1}, []) == []


class TestAlerters:
    def test_console_alerter_writes_to_stream(self):
        import io

        stream = io.StringIO()
        ConsoleAlerter(stream=stream).send([Alert("m", ">", 1, 5)])
        assert "m" in stream.getvalue()

    def test_build_alerters_always_includes_console(self):
        """A console alerter is always present, even with no config."""
        built = build_alerters(None)
        assert len(built) == 1
        assert isinstance(built[0], ConsoleAlerter)

    def test_build_alerters_adds_webhook(self):
        built = build_alerters({"url_env": "TEST_WEBHOOK_URL"})
        assert len(built) == 2
        assert any(isinstance(a, WebhookAlerter) for a in built)

    def test_webhook_alerter_uses_injected_urlopen(self, monkeypatch):
        monkeypatch.setenv("TEST_WEBHOOK_URL", "https://hooks.example.com/abc")
        sent = {}

        class FakeResponse:
            def read(self):
                return b"ok"

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        def fake_urlopen(request, timeout=None):
            sent["url"] = request.full_url
            sent["data"] = request.data
            sent["timeout"] = timeout
            return FakeResponse()

        alerter = WebhookAlerter(url_env="TEST_WEBHOOK_URL", urlopen=fake_urlopen)
        alerter.send([Alert("m", ">", 1, 5)])
        assert sent["url"] == "https://hooks.example.com/abc"
        assert b"m" in sent["data"]

    def test_webhook_alerter_without_env_is_inert(self):
        import os

        os.environ.pop("DRIFT_SENTINEL_TEST_ABSENT", None)
        alerter = WebhookAlerter(url_env="DRIFT_SENTINEL_TEST_ABSENT")
        assert alerter._resolve_url() is None

    def test_webhook_payload_shape(self):
        alerter = WebhookAlerter(url_env="X")
        payload = alerter.build_payload([Alert("m", ">", 1, 5)])
        assert isinstance(payload, dict)
        assert payload


# --------------------------------------------------------------------------
# csv io
# --------------------------------------------------------------------------

class TestCsvIo:
    def test_round_trip(self, tmp_path):
        rows = [{"a": 1.0, "b": "x"}, {"a": 2.0, "b": "y"}]
        path = tmp_path / "d.csv"
        write_csv(rows, str(path))
        assert read_csv(str(path)) == rows

    def test_numeric_coercion(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("a\n1.5\n2.5\n", encoding="utf-8")
        assert read_csv(str(path)) == [{"a": 1.5}, {"a": 2.5}]

    def test_categorical_stays_text(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("a\nx\ny\n", encoding="utf-8")
        assert read_csv(str(path)) == [{"a": "x"}, {"a": "y"}]

    def test_empty_numeric_cell_becomes_zero(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("a,b\n1.0,\n2.0,4.0\n", encoding="utf-8")
        rows = read_csv(str(path))
        assert rows[0]["b"] == 0.0

    def test_empty_categorical_cell_stays_empty_string(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("a\nx\n\ny\n", encoding="utf-8")
        # A blank line is not a record for csv.DictReader, so it is skipped
        # rather than read as an all-empty row.
        assert read_csv(str(path)) == [{"a": "x"}, {"a": "y"}]

    def test_missing_header_raises(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            read_csv(str(path))

    def test_mixed_column_is_categorical(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("a\n1.0\nn/a\n", encoding="utf-8")
        assert read_csv(str(path))[1]["a"] == "n/a"

    def test_write_empty_rows(self, tmp_path):
        path = tmp_path / "d.csv"
        write_csv([], str(path))
        assert path.read_text(encoding="utf-8") == ""

    def test_write_unions_columns(self, tmp_path):
        path = tmp_path / "d.csv"
        write_csv([{"a": 1}, {"b": 2}], str(path))
        assert path.read_text(encoding="utf-8").splitlines()[0] == "a,b"


class TestSplitColumns:
    def test_classifies(self):
        rows = [{"n": 1.0, "c": "x"}, {"n": 2.0, "c": "y"}]
        split = split_numeric_categorical(rows)
        assert split["numeric"] == ["n"]
        assert split["categorical"] == ["c"]

    def test_empty_input(self):
        assert split_numeric_categorical([]) == {"numeric": [], "categorical": []}

    def test_all_empty_column_is_categorical(self):
        assert split_numeric_categorical([{"a": ""}])["categorical"] == ["a"]


# --------------------------------------------------------------------------
# report + prometheus
# --------------------------------------------------------------------------

class TestReport:
    def test_json_report_shape(self):
        report = build_json_report({"verdict": "stable"}, alerts=[], meta={"a": 1})
        assert set(report) == {"generated_at", "meta", "alerts", "metrics"}

    def test_markdown_renders_drift_features(self, normal_reference, shifted_current):
        """Regression: the renderer once ignored the features list entirely."""
        metrics = compare_datasets({"age": normal_reference},
                                   {"age": shifted_current}).to_dict()
        md = build_markdown_report(build_json_report(metrics), title="T")
        assert "## Feature drift" in md
        assert "age" in md
        assert "significant_drift" in md

    def test_markdown_renders_verdict(self, normal_reference):
        metrics = compare_datasets({"age": normal_reference},
                                   {"age": list(normal_reference)}).to_dict()
        md = build_markdown_report(build_json_report(metrics))
        assert "stable" in md

    def test_markdown_renders_llm_metrics(self, llm_entries):
        md = build_markdown_report(build_json_report(compute_llm_metrics(llm_entries).to_dict()))
        assert "## LLM metrics" in md
        assert "Refusal rate" in md

    def test_markdown_renders_alerts(self):
        md = build_markdown_report(
            build_json_report({"a": 1}, alerts=[{"metric": "a", "operator": ">", "threshold": 0, "actual": 1}])
        )
        assert "## Alerts" in md

    def test_markdown_handles_empty_metrics(self):
        assert build_markdown_report(build_json_report({}))

    def test_legacy_feature_drift_shape_still_renders(self):
        md = build_markdown_report(
            build_json_report({"feature_drift": {"a": {"type": "numeric", "psi": 0.2, "band": "moderate"}}})
        )
        assert "## Feature drift" in md


class TestPrometheus:
    def test_basic_gauge(self):
        text = render_metrics({"drift_age_psi": 0.5})
        assert "drift_age_psi 0.5" in text
        assert "# TYPE drift_age_psi gauge" in text

    def test_nested_dicts_are_flattened(self):
        assert "latency_ms_p90 12.0" in render_metrics({"latency_ms": {"p90": 12.0}})

    def test_none_values_skipped(self):
        assert "x" not in render_metrics({"x": None})

    def test_booleans_render_as_one_and_zero(self):
        text = render_metrics({"yes": True, "no": False})
        assert "yes 1" in text and "no 0" in text

    def test_strings_are_skipped(self):
        """Regression: a gauge sample must be numeric."""
        text = render_metrics({"verdict": "significant_drift"})
        assert text == ""

    def test_feature_list_becomes_one_gauge_per_feature(self, normal_reference, shifted_current):
        """Regression: lists used to be dropped entirely."""
        metrics = compare_datasets({"age": normal_reference},
                                   {"age": shifted_current}).to_dict()
        text = render_metrics(metrics)
        assert "drift_age_psi" in text
        assert "significant_drift" not in text

    def test_illegal_characters_are_escaped(self):
        text = render_metrics({"a b-c": 1.0})
        assert "a_b_c 1.0" in text

    def test_leading_digit_gets_underscore(self):
        assert "x_1y" in render_metrics({"x-1y": 2.0})

    def test_requires_dict(self):
        with pytest.raises(TypeError):
            render_metrics([1, 2])
