"""End-to-end tour of drift-sentinel. Runs fully offline.

    python examples/demo_end_to_end.py

Compares two generated CSV datasets, scores an LLM log, fires alert rules,
and renders a Prometheus exposition. Nothing here touches the network.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drift_sentinel.alerts import evaluate_alert_rules  # noqa: E402
from drift_sentinel.drift import compare_datasets  # noqa: E402
from drift_sentinel.llm_monitors import compute_llm_metrics  # noqa: E402
from drift_sentinel.prometheus import render_metrics  # noqa: E402
from drift_sentinel.report import build_json_report, build_markdown_report  # noqa: E402
from drift_sentinel.stats import js_distance, ks_pvalue, ks_statistic, psi  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="drift-demo-"))

    # -- 1. Statistical primitives -----------------------------------------
    rule("STEP 1 - DRIFT STATISTICS")
    generator = random.Random(42)
    reference = [generator.gauss(0.0, 1.0) for _ in range(1000)]
    shifted = [generator.gauss(1.0, 1.0) for _ in range(1000)]
    same = list(reference)

    print(f"PSI(identical)            = {psi(reference, same):.6f}")
    print(f"PSI(shifted by 1 sd)      = {psi(reference, shifted):.4f}")
    print(f"KS statistic              = {ks_statistic(reference, shifted):.4f}")
    print(f"KS p-value                = {ks_pvalue(ks_statistic(reference, shifted), 1000, 1000):.3g}")
    print(f"JS distance (categories)  = "
          f"{js_distance(['a'] * 100 + ['b'] * 100, ['a'] * 95 + ['b'] * 5):.4f}")

    # -- 2. Dataset comparison --------------------------------------------
    rule("STEP 2 - DATASET COMPARISON")
    reference_columns = {
        "age": reference,
        "plan": [generator.choice(["free", "pro", "team"]) for _ in range(1000)],
        "region": [generator.choice(["emea", "amer", "apac"]) for _ in range(1000)],
    }
    current_columns = {
        "age": shifted,
        "plan": [generator.choice(["free", "pro"]) for _ in range(1000)],
        "region": [generator.choice(["apac", "amer"]) for _ in range(1000)],
    }
    report = compare_datasets(reference_columns, current_columns)
    print(f"verdict: {report.verdict}")
    print()
    print(f"{'FEATURE':<10} {'KIND':<12} {'METRIC':<12} {'VALUE':>8}  BAND")
    for feature in report.features:
        print(f"{feature.feature:<10} {feature.kind:<12} {feature.metric:<12} "
              f"{feature.value:>8.4f}  {feature.band}")

    # -- 3. Alerting ------------------------------------------------------
    rule("STEP 3 - ALERT RULES")
    rules = json.loads((REPO_ROOT / "examples" / "alert_rules.json").read_text(encoding="utf-8"))
    try:
        fired = evaluate_alert_rules(report.to_dict(), [
            {"metric": "verdict", "operator": "!=", "threshold": "stable"}
        ])
        for alert in fired:
            print(f"  FIRED: {alert.format_console()}")
    except KeyError as exc:
        print(f"  (no drift rules fired: {exc})")
    print(f"  configured LLM rules: {len(rules)} (applied to the LLM log below)")

    # -- 4. LLM monitoring -------------------------------------------------
    rule("STEP 4 - LLM CALL MONITORING")
    entries = []
    healthy = "Restart the ingestion worker; the queue drains within five minutes."
    for i in range(200):
        refusing = 120 <= i < 180          # a burst of refusals
        entries.append({
            "model": "demo-model",
            "prompt_tokens": generator.randint(150, 900),
            "completion_tokens": generator.randint(30, 200),
            "latency_ms": round(generator.gauss(1900 if refusing else 800, 150), 1),
            "response": "I'm sorry, I can't help with that." if refusing else healthy,
        })
    metrics = compute_llm_metrics(entries, reference_response=healthy)
    payload = metrics.to_dict()
    for key in ("entry_count", "refusal_rate", "total_prompt_tokens", "total_cost"):
        print(f"  {key:<22} {payload[key]}")
    print(f"  latency p50/p90/p99     {payload['latency_ms']['p50']:.0f} / "
          f"{payload['latency_ms']['p90']:.0f} / {payload['latency_ms']['p99']:.0f} ms")

    llm_alerts = evaluate_alert_rules(payload, rules)
    print()
    print(f"  {len(llm_alerts)} of {len(rules)} rules fired:")
    for alert in llm_alerts:
        print(f"    {alert.format_console()}")

    # -- 5. Reports and export --------------------------------------------
    rule("STEP 5 - REPORT RENDERING")
    document = build_json_report(payload, alerts=[a.to_dict() for a in llm_alerts],
                                 meta={"demo": True})
    markdown = build_markdown_report(document, title="Demo monitoring report")
    print(f"Markdown report: {len(markdown.splitlines())} lines, "
          f"{len(document['metrics'])} metric groups")
    report_path = workdir / "report.json"
    report_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"JSON report written to {report_path}")

    rule("STEP 6 - PROMETHEUS EXPOSITION")
    exposition = render_metrics(report.to_dict())
    print(exposition.rstrip())

    rule("DEMO COMPLETE")
    print(f"artefacts written to: {workdir}")
    print("Everything above ran offline: no network, no API keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
