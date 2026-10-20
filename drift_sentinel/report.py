"""JSON and Markdown report builders.

Reports are assembled from plain dictionaries so a caller can render a
metric source without knowing about the underlying classes.

Reports are rendered from plain metric dictionaries (the ``to_dict`` output
of :class:`drift_sentinel.drift.DriftReport` or
:class:`drift_sentinel.llm_monitors.LLMMetrics`), so reports work for any
metric source without knowing about the underlying classes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "build_json_report",
    "build_markdown_report",
    "save_json_report",
    "save_markdown_report",
]


def _utc_timestamp() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``.

Uses timezone-aware ``datetime.now(timezone.utc)`` rather than the deprecated

    Uses a timezone-aware ``now(timezone.utc)`` rather than the deprecated
    ``datetime.utcnow()``, which is scheduled for removal in Python 3.12+.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_json_report(
    metrics: Dict[str, Any],
    alerts: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical JSON report mapping.

    Args:
        metrics: Metric dictionary (drift or LLM metrics ``to_dict`` output).
        alerts: Fired alerts as serialized dictionaries.
        meta: Optional run metadata (reference path, current path, model
            names, CLI version, etc.).

    Returns:
        A JSON-compatible mapping with sorted keys, ready to be written as JSON.
    """
    return {
        "generated_at": _utc_timestamp(),
        "meta": meta or {},
        "alerts": alerts or [],
        "metrics": metrics,
    }


def save_json_report(
    report: Dict[str, Any],
    path: str,
) -> str:
    """Write a JSON report to disk.

    Args:
        report: Report dictionary from :func:`build_json_report`.
        path: Destination file path.

    Returns:
        The same ``path`` for convenient chaining.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def _format_scalar(value: Any) -> str:
    """Render a scalar metric value for Markdown tables."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)


def _render_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    """Render a Markdown table as a list of lines."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_scalar(cell) for cell in row) + " |")
    return lines


def _render_drift_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Render the feature-drift rows as Markdown table lines.

    Two shapes are supported:

    The renderer consumes plain dictionaries as well as with real comparison output...

    1. The canonical shape produced by :meth:`DriftReport.to_dict` -
       ``{"verdict": str, "features": [{"feature", "kind", "metric",
       "value", "band"}, ...]}``.
    2. A legacy shape - ``{"feature_drift": {name: {...}}}`` plus an
       optional ``{"overall": {...}}`` block.

    Supporting both keeps the renderer usable with hand-built metric
    dictionaries as well as with real comparison output.
    """
    lines: List[str] = []

    features = metrics.get("features")
    if isinstance(features, list) and features:
        verdict = metrics.get("verdict")
        if verdict is not None:
            lines.append(f"**Verdict:** `{verdict}`")
            lines.append("")
        lines.append("## Feature drift")
        lines.append("")
        rows: List[List[Any]] = []
        for entry in features:
            if not isinstance(entry, dict):
                continue
            rows.append(
                [
                    entry.get("feature"),
                    entry.get("kind"),
                    entry.get("metric"),
                    entry.get("value"),
                    entry.get("band"),
                    entry.get("pvalue"),
                ]
            )
        lines.extend(
            _render_table(
                ["Feature", "Type", "Metric", "Value", "Band", "p-value"], rows
            )
        )
        lines.append("")

    feature_drift = metrics.get("feature_drift")
    if isinstance(feature_drift, dict) and feature_drift:
        lines.append("## Feature drift")
        lines.append("")
        headers = ["Feature", "Type", "PSI", "Band", "KS", "KS p-value", "JS distance"]
        rows: List[List[Any]] = []
        for name, stats in feature_drift.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                [
                    name,
                    stats.get("type", "-"),
                    stats.get("psi"),
                    stats.get("band", "-"),
                    stats.get("ks"),
                    stats.get("ks_pvalue"),
                    stats.get("js"),
                ]
            )
        lines.extend(_render_table(headers, rows))
        lines.append("")
    overall = metrics.get("overall")
    if isinstance(overall, dict) and overall:
        lines.append("## Overall summary")
        lines.append("")
        lines.extend(
            _render_table(
                ["Metric", "Value"],
                [[key, value] for key, value in sorted(overall.items())],
            )
        )
        lines.append("")
    return lines


def _render_llm_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Render LLM-style metrics sections."""
    lines: List[str] = []
    scalar_keys = [
        ("entry_count", "Entries analysed"),
        ("refusal_rate", "Refusal rate"),
        ("total_prompt_tokens", "Total prompt tokens"),
        ("total_completion_tokens", "Total completion tokens"),
        ("total_cost", "Total cost"),
        ("jaccard_similarity", "Jaccard similarity vs reference"),
    ]
    known = [item for item in scalar_keys if item[0] in metrics]
    if known:
        lines.append("## LLM metrics")
        lines.append("")
        lines.extend(
            _render_table(["Metric", "Value"], [[label, metrics[key]] for key, label in known])
        )
        lines.append("")
    latency = metrics.get("latency_ms")
    if isinstance(latency, dict):
        lines.append("### Latency (ms)")
        lines.append("")
        lines.extend(
            _render_table(
                ["Count", "Mean", "p50", "p90", "p99", "Max"],
                [[latency.get(k) for k in ("count", "mean", "p50", "p90", "p99", "max")]],
            )
        )
        lines.append("")
    response_length = metrics.get("response_length")
    if isinstance(response_length, dict):
        lines.append("### Response length")
        lines.append("")
        lines.extend(
            _render_table(
                ["Count", "Mean", "p50", "p90", "p99", "Max"],
                [
                    [
                        response_length.get(k)
                        for k in ("count", "mean", "p50", "p90", "p99", "max")
                    ]
                ],
            )
        )
        lines.append("")
    per_model = metrics.get("per_model")
    if isinstance(per_model, dict) and per_model:
        lines.append("### Per model")
        lines.append("")
        headers = ["Model", "Entries", "Prompt tokens", "Completion tokens", "Cost"]
        rows = []
        for model, bucket in sorted(per_model.items()):
            if not isinstance(bucket, dict):
                continue
            rows.append(
                [
                    model,
                    bucket.get("entry_count"),
                    bucket.get("prompt_tokens"),
                    bucket.get("completion_tokens"),
                    bucket.get("cost"),
                ]
            )
        lines.extend(_render_table(headers, rows))
        lines.append("")
    return lines


def _looks_like_llm_metrics(metrics: Dict[str, Any]) -> bool:
    """Heuristically decide whether a metrics dict is LLM-style output."""
    return any(key in metrics for key in ("entry_count", "latency_ms", "refusal_rate"))


def _render_alerts(alerts: List[Dict[str, Any]]) -> List[str]:
    """Render the fired-alerts section."""
    lines: List[str] = []
    if alerts:
        lines.append("## Alerts")
        lines.append("")
        for alert in alerts:
            line = "- **{} {} {}** (actual: {})".format(
                alert.get("metric", "?"),
                alert.get("operator", "?"),
                _format_scalar(alert.get("threshold")),
                _format_scalar(alert.get("actual")),
            )
            message = alert.get("message")
            if message:
                line += " - {}".format(message)
            lines.append(line)
        lines.append("")
    else:
        lines.append("## Alerts")
        lines.append("")
        lines.append("No alerts fired.")
        lines.append("")
    return lines


def build_markdown_report(
    report: Dict[str, Any],
    title: str = "drift-sentinel report",
) -> str:
    """Render a metrics report as Markdown.

    Args:
        report: Report dictionary from :func:`build_json_report` (or loaded
            from a previously saved JSON report file).
        title: Top-level document title.

    Returns:
        Complete Markdown text for the report.
    """
    metrics = report.get("metrics", {})
    alerts = report.get("alerts", [])
    meta = report.get("meta", {})

    lines: List[str] = ["# {}".format(title), ""]
    lines.append("Generated at: {}".format(report.get("generated_at", _utc_timestamp())))
    lines.append("")
    if meta:
        lines.extend(
            _render_table(
                ["Field", "Value"], [[key, value] for key, value in sorted(meta.items())]
            )
        )
        lines.append("")

    if _looks_like_llm_metrics(metrics):
        lines.extend(_render_llm_metrics(metrics))
    else:
        lines.extend(_render_drift_metrics(metrics))
        lines.extend(_render_llm_metrics({k: v for k, v in metrics.items()
                                          if k in ("llm",) and isinstance(v, dict)}))

    lines.extend(_render_alerts(alerts))
    return "\n".join(lines)


def save_markdown_report(markdown: str, path: str) -> str:
    """Write Markdown to disk.

    Args:
        markdown: Rendered Markdown text from :func:`build_markdown_report`.
        path: Destination file path.

    Returns:
        The same ``path`` for convenient chaining.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
        if not markdown.endswith("\n"):
            handle.write("\n")
    return path
