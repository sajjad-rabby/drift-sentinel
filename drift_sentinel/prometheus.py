"""Prometheus exposition format and a scrape server.

Public helpers:
- render_metrics: exposition text for a metrics dictionary
- serve_metrics: blocking HTTP server on GET /metrics
    - render_metrics: dict -> Prometheus text body
    - serve_metrics: blocking HTTP server on GET /metrics
"""

import http.server
import re
import socketserver
import threading
from typing import Dict, Optional

__all__ = ["render_metrics", "serve_metrics"]

# Legal characters in a Prometheus metric name (after the required prefix).
_METRIC_NAME_LEGAL = re.compile(r"[^a-zA-Z0-9:]")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 9090


def _escape_metric_name(name: str) -> str:
    """Coerce a string into a legal Prometheus metric name.

Args:
    name: Raw name; characters outside ``[a-zA-Z0-9_:]`` become underscores
        (a leading digit is prefixed with ``_``).

    Dots become underscores; any other character outside ``[a-zA-Z0-9:]``
    becomes an underscore as well. The result always starts with a letter
    or underscore (Prometheus requires the first character of a metric
    name to be ``[a-zA-Z_:]``).
    """
    sanitized = _METRIC_NAME_LEGAL.sub("_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_unnamed"


def _format_value(value) -> str:
    """Format a numeric value for the exposition body."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return repr(float(value)) if isinstance(value, float) else str(value)


def _flatten(metrics: Dict, prefix: str = "") -> Dict[str, object]:
    """Flatten nested dictionaries into dotted keys, skipping non-numeric leaves.

Returns:
    Mapping of dotted key to numeric value. Non-numeric leaves are dropped because
    ``str([{...}])`` would produce a sample containing a Python list rather than a number.

    Lists are skipped: a Prometheus gauge needs a scalar, and emitting
    ``str([{...}])`` would produce a sample containing a Python repr.
    """
    flat: Dict[str, object] = {}
    for key, value in metrics.items():
        full_key = "{}.{}".format(prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        elif isinstance(value, list):
            continue
        else:
            flat[full_key] = value
    return flat


def _expand_features(metrics: Dict) -> Dict[str, object]:
    """Turn a drift report into one metric per feature.

A monitoring

    ``{"verdict": ..., "features": [{"feature": "age", "metric": "psi",
    "value": 0.23, ...}]}`` becomes ``drift_age_psi`` -> 0.23. A monitoring
    system needs one time series per feature, not a Python repr of a list.
    """
    extra: Dict[str, object] = {}
    features = metrics.get("features")
    if not isinstance(features, list):
        return extra
    for entry in features:
        if not isinstance(entry, dict):
            continue
        feature = entry.get("feature")
        if not isinstance(feature, str) or not feature:
            continue
        value = entry.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        suffix = entry.get("metric")
        key = "drift_{}_{}".format(feature, suffix) if suffix else "drift_{}".format(feature)
        extra[key] = float(value)
    return extra


def render_metrics(metrics: Dict) -> str:
    """Render a metrics dictionary as Prometheus exposition text.

Each numeric leaf becomes its own gauge; in particular a drift report yields one
gauge per feature.

    Every numeric leaf becomes a gauge with ``# HELP`` and ``# TYPE`` lines.
    ``None`` values are skipped, booleans render as ``1``/``0``, and
    non-numeric strings (such as a drift ``verdict``) are skipped because a
    gauge sample must be a number. Drift feature lists are expanded into one
    gauge per feature.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    lines = []
    for dotted_key, value in sorted(_flatten(metrics).items()):
        if value is None:
            continue
        if not isinstance(value, (int, float, bool)):
            continue  # a gauge sample must be numeric
        name = _escape_metric_name(dotted_key)
        label = dotted_key.replace("_", " ").replace(".", " - ")
        lines.append("# HELP {} {}".format(name, label))
        lines.append("# TYPE {} gauge".format(name))
        lines.append("{} {}".format(name, _format_value(value)))

    for dotted_key, value in sorted(_expand_features(metrics).items()):
        name = _escape_metric_name(dotted_key)
        label = dotted_key.replace("_", " ").replace(".", " - ")
        if name not in {line.split(" ", 1)[1] for line in lines if line.startswith("# TYPE ")}:
            lines.append("# HELP {} {}".format(name, label))
            lines.append("# TYPE {} gauge".format(name))
        lines.append("{} {}".format(name, _format_value(value)))

    return "\n".join(lines) + ("\n" if lines else "")


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler serving the rendered metrics on GET /metrics."""

    metrics: Dict = {}
    lock = threading.Lock()

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/metrics":
            self.send_error(404, "Not Found")
            return
        with _MetricsHandler.lock:
            body = render_metrics(_MetricsHandler.metrics).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quiet by default; the CLI already reports the listening address.
        return


def serve_metrics(
    metrics: Dict,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> None:
    """Serve the exposition endpoint until interrupted.

Args:
    metrics: Metrics dictionary to expose.
    host: Interface to bind; loopback by default.
    port: Port to bind.

The server responds

    Blocks the calling thread; stop with Ctrl+C. Serves only GET requests
    to ``/metrics``; every other path returns 404.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    with _MetricsHandler.lock:
        _MetricsHandler.metrics = dict(metrics)
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((host, port), _MetricsHandler)
    try:
        print("drift-sentinel metrics listening on http://{}:{}/metrics".format(host, port))
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
