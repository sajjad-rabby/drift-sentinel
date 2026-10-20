"""Command-line interface for drift-sentinel.

Everything runs offline except ``serve`` (which only binds a local socket).
    compare    compare two CSV datasets column by column
    llm        score a JSONL log of LLM calls
    report     re-render a saved JSON report as Markdown
    serve      expose the last report on a Prometheus scrape endpoint

Everything runs offline except ``serve`` (which only binds a local socket).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .alerts import build_alerters, evaluate_alert_rules
from .csv_io import read_csv, split_numeric_categorical
from .drift import compare_datasets
from .llm_monitors import compute_llm_metrics
from .prometheus import serve_metrics
from .report import (
    build_json_report,
    build_markdown_report,
    save_json_report,
    save_markdown_report,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DRIFT = 2


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_ERROR


def _columns_to_dict(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Transpose row dicts into column dicts for compare_datasets."""
    columns: Dict[str, List[Any]] = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def _emit(text: str, out: Optional[str]) -> None:
    if out is None or out == "-":
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    else:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")


def _load_alert_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("alerts", [])
    if not isinstance(data, list):
        raise ValueError("alert config must be a JSON array of rule objects")
    return data


def _load_webhook_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("webhook config must be a JSON object")
    return data


def _exit_code_for(verdict: str, fail_on: Optional[str]) -> int:
    """Map a verdict to a process exit code for CI use."""
    if fail_on is None or verdict == "no_data":
        return EXIT_OK
    order = ["stable", "moderate_drift", "significant_drift"]
    try:
        return EXIT_DRIFT if order.index(verdict) >= order.index(fail_on) else EXIT_OK
    except ValueError:
        return EXIT_ERROR


def cmd_compare(args: argparse.Namespace) -> int:

    """Implement cmd compare.
    Returns:
        Process exit code: 0 normally, 2 when the fail-on gate is tripped.
    """
    try:
        ref_rows = read_csv(args.reference)
        cur_rows = read_csv(args.current)
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    reference = _columns_to_dict(ref_rows)
    current = _columns_to_dict(cur_rows)

    try:
        alert_configs = _load_alert_config(args.alerts)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(f"alert config: {exc}")

    report = compare_datasets(reference, current, bins=args.bins)
    metrics = report.to_dict()

    alerts = []
    if alert_configs:
        try:
            alerts = evaluate_alert_rules(metrics, alert_configs)
        except (ValueError, KeyError) as exc:
            return _fail(f"alert rules: {exc}")

    webhook = None
    if args.webhook_config:
        try:
            webhook = _load_webhook_config(args.webhook_config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _fail(f"webhook config: {exc}")

    meta = {
        "reference": str(args.reference),
        "current": str(args.current),
        "reference_rows": len(ref_rows),
        "current_rows": len(cur_rows),
        "bins": args.bins,
    }
    # Alerters need Alert objects; the report needs their dict form.
    fired = [a.to_dict() for a in alerts]
    document = build_json_report(metrics, alerts=fired, meta=meta)

    if args.json:
        _emit(json.dumps(document, indent=2), args.output)
    else:
        _emit(build_markdown_report(document, title=args.title), args.output)

    if alerts:
        for alerter in build_alerters(webhook):
            alerter.send(alerts)

    return _exit_code_for(report.verdict, args.fail_on)


def cmd_llm(args: argparse.Namespace) -> int:
    """Process an LLM log file and emit a drift report.

    Returns:
        Process exit code; non-zero when the log cannot be processed.
    """
    path = Path(args.log)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return _fail(str(exc))

    entries: List[Dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            return _fail(f"line {lineno}: invalid JSON ({exc})")
        if not isinstance(entry, dict):
            return _fail(f"line {lineno}: expected a JSON object")
        entries.append(entry)

    if not entries:
        return _fail("log contains no entries")

    price_table: Optional[Dict[str, Dict[str, float]]] = None
    if args.price_table:
        try:
            price_table = json.loads(Path(args.price_table).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _fail(f"price table: {exc}")

    reference = None
    if args.reference:
        ref_path = Path(args.reference)
        try:
            reference = ref_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            return _fail(f"--reference must be a readable file: {exc}")

    metrics = compute_llm_metrics(entries, reference_response=reference,
                                  price_table=price_table)
    payload = metrics.to_dict()

    try:
        alert_configs = _load_alert_config(args.alerts)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(f"alert config: {exc}")

    alerts = []
    if alert_configs:
        try:
            alerts = evaluate_alert_rules(payload, alert_configs)
        except (ValueError, KeyError) as exc:
            return _fail(f"alert rules: {exc}")

    fired = [a.to_dict() for a in alerts]
    document = build_json_report(
        payload,
        alerts=fired,
        meta={"log": str(path), "entries": len(entries)},
    )

    if args.json:
        _emit(json.dumps(document, indent=2), args.output)
    else:
        _emit(build_markdown_report(document, title=args.title), args.output)

    if alerts and args.fail_on_alert:
        return EXIT_DRIFT
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:

    """Implement cmd report.
    Returns:
        Process exit code; non-zero when the report cannot be read.
    """
    try:
        document = json.loads(Path(args.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(str(exc))
    _emit(build_markdown_report(document, title=args.title), args.output)
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:

    """Implement cmd serve.
    Returns:
        Process exit code; non-zero when the report cannot be served.
    """
    try:
        metrics = json.loads(Path(args.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(str(exc))

    body = metrics.get("metrics", metrics)
    print(f"serving metrics on http://{args.host}:{args.port}/metrics")
    try:
        serve_metrics(body, host=args.host, port=args.port)
    except OSError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        print("\nstopped")
    return EXIT_OK


def cmd_columns(args: argparse.Namespace) -> int:

    """Implement cmd columns.
    Returns:
        Process exit code: 0 on success.
    """
    try:
        rows = read_csv(args.path)
    except (OSError, ValueError) as exc:
        return _fail(str(exc))
    split = split_numeric_categorical(rows)
    print(f"{args.path}: {len(rows)} row(s)")
    print()
    print("numeric     :", ", ".join(split["numeric"]) or "(none)")
    print("categorical :", ", ".join(split["categorical"]) or "(none)")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the full ``drift-sentinel`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="drift-sentinel",
        description="Production drift and LLM monitoring: PSI, KS, JS distance, "
        "alerting, Prometheus export.",
    )
    parser.add_argument(
        "--version", action="version", version=f"drift-sentinel {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    compare = sub.add_parser("compare", help="compare two CSV datasets")
    compare.add_argument("--reference", required=True, help="baseline CSV")
    compare.add_argument("--current", required=True, help="current CSV")
    compare.add_argument("--bins", type=int, default=10, help="PSI quantile bins")
    compare.add_argument("--json", action="store_true", help="emit JSON, not Markdown")
    compare.add_argument("--output", help="output file ('-' for stdout)")
    compare.add_argument("--title", default="drift-sentinel report", help="Markdown title")
    compare.add_argument("--alerts", help="JSON file of alert rules")
    compare.add_argument("--webhook-config", help="JSON file with webhook settings")
    compare.add_argument(
        "--fail-on",
        choices=("stable", "moderate_drift", "significant_drift"),
        help="exit 2 when the verdict is at least this severe (for CI)",
    )
    compare.set_defaults(func=cmd_compare)

    llm = sub.add_parser("llm", help="score a JSONL log of LLM calls")
    llm.add_argument("--log", required=True, help="JSONL log file")
    llm.add_argument("--reference", help="reference response text for Jaccard scoring")
    llm.add_argument("--price-table", help="JSON model price table for cost estimation")
    llm.add_argument("--json", action="store_true", help="emit JSON, not Markdown")
    llm.add_argument("--output", help="output file ('-' for stdout)")
    llm.add_argument("--title", default="drift-sentinel LLM report")
    llm.add_argument("--alerts", help="JSON file of alert rules")
    llm.add_argument("--fail-on-alert", action="store_true", help="exit 2 if any alert fires")
    llm.set_defaults(func=cmd_llm)

    report = sub.add_parser("report", help="re-render a JSON report as Markdown")
    report.add_argument("--path", required=True, help="saved JSON report")
    report.add_argument("--output", help="output file ('-' for stdout)")
    report.add_argument("--title", default="drift-sentinel report")
    report.set_defaults(func=cmd_report)

    serve = sub.add_parser("serve", help="serve saved metrics for Prometheus")
    serve.add_argument("--path", required=True, help="saved JSON report or metrics")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9108)
    serve.set_defaults(func=cmd_serve)

    columns = sub.add_parser("columns", help="show detected column types")
    columns.add_argument("--path", required=True, help="CSV file")
    columns.set_defaults(func=cmd_columns)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
