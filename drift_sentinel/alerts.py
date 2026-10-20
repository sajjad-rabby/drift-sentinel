"""Threshold alerts over a metrics dictionary.

Metrics are addressed with dotted paths, so a nested metric such as
``metrics["feature_drift"]["income"]["psi"]`` is written ``feature_drift.income.psi``.

.. code-block:: json

    {
      "metric": "feature_drift.income.psi",
      "operator": ">",
      "threshold": 0.25,
      "message": "Income feature PSI exceeded significant band"
    }

Metric lookup supports dotted keys into nested metric dictionaries (e.g.
``feature_drift.income.psi`` resolves to
``metrics["feature_drift"]["income"]["psi"]``).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "AlertRule",
    "Alert",
    "Alerter",
    "ConsoleAlerter",
    "WebhookAlerter",
    "ALERT_OPERATORS",
    "get_operator",
    "evaluate_operator",
    "lookup_metric",
    "evaluate_alert_rules",
]


#: Supported comparison operators keyed by their textual representation.
ALERT_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    ">": lambda actual, threshold: actual > threshold,
    ">=": lambda actual, threshold: actual >= threshold,
    "<": lambda actual, threshold: actual < threshold,
    "<=": lambda actual, threshold: actual <= threshold,
    "==": lambda actual, threshold: actual == threshold,
    "!=": lambda actual, threshold: actual != threshold,
}


def get_operator(operator: str) -> Callable[[Any, Any], bool]:
    """Return the comparison function for an operator string.

    Args:
        operator: One of ``>, >=, <, <=, ==, !=``.

    Returns:
        A two-argument callable ``(actual, threshold) -> bool``.

    Raises:
        ValueError: If the operator string is not recognised.
    """
    try:
        return ALERT_OPERATORS[operator]
    except KeyError:
        supported = ", ".join(sorted(ALERT_OPERATORS))
        raise ValueError(
            "Unsupported alert operator {!r}; supported operators: {}".format(
                operator, supported
            )
        ) from None


def evaluate_operator(actual: Any, operator: str, threshold: Any) -> bool:
    """Evaluate ``actual <operator> threshold`` using the named operator."""
    return get_operator(operator)(actual, threshold)


def lookup_metric(metrics: Dict[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted metric path against a nested dictionary.

    Args:
        metrics: Nested dictionary of metric values.
        dotted_key: Dotted path such as ``"feature_drift.income.psi"``.

    Returns:
        The value found at the path.

    Raises:
        KeyError: If any segment of the path is missing along the way.
    """
    node: Any = metrics
    for segment in dotted_key.split("."):
        if not isinstance(node, dict) or segment not in node:
            raise KeyError("Metric path {!r} not found in metrics".format(dotted_key))
        node = node[segment]
    return node


class AlertRule:
    """A single threshold-based alert rule loaded from configuration."""

    def __init__(
        self,
        metric: str,
        operator: str,
        threshold: Any,
        message: Optional[str] = None,
    ) -> None:
        """Create a rule.

        Args:
            metric: Dotted metric path to evaluate (e.g. ``"overall.ks_pvalue"``).
            operator: Comparison operator string.
            threshold: Value the metric is compared against.
            message: Optional human-readable message included in fired alerts.

        Raises:
            ValueError: If ``operator`` is not one of the supported operators.
        """
        self.metric = metric
        self.operator = operator
        self.threshold = threshold
        self.message = message
        # Validate eagerly so misconfigured rules fail at load time, not fire time.
        get_operator(operator)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AlertRule":
        """Build an :class:`AlertRule` from a config dictionary."""
        return cls(
            metric=str(raw["metric"]),
            operator=str(raw["operator"]),
            threshold=raw["threshold"],
            message=raw.get("message"),
        )

    def check(self, metrics: Dict[str, Any]) -> Optional["Alert"]:

        """Evaluate the rule against a metrics dictionary.

Returns:
    The :class:`Alert` when the rule
            An :class:`Alert` if the metric is present and the comparison
            fires, otherwise ``None`` (rule not triggered, or metric missing).
        """
        try:
            actual = lookup_metric(metrics, self.metric)
        except KeyError:
            return None
        if actual is None:
            return None
        if evaluate_operator(actual, self.operator, self.threshold):
            return Alert(
                metric=self.metric,
                operator=self.operator,
                threshold=self.threshold,
                actual=actual,
                message=self.message or "{} {} {}".format(self.metric, self.operator, self.threshold),
            )
        return None


class Alert:
    """A fired alert with full provenance of what triggered it."""

    def __init__(
        self,
        metric: str,
        operator: str,
        threshold: Any,
        actual: Any,
        message: Optional[str] = None,
    ) -> None:

        """Create an alert.

Args:
    metric: Dotted metric path that triggered.
    operator: Comparison that fired.
    threshold: Configured threshold.
    actual: Observed value.
    message: Optional human-readable message.
            metric: Dotted metric path that fired.
            operator: Operator used in the comparison.
            threshold: Threshold value from the rule.
            actual: Observed metric value at evaluation time.
            message: Optional human-readable message.
        """
        self.metric = metric
        self.operator = operator
        self.threshold = threshold
        self.actual = actual
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the alert for reports and webhook payloads."""
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "actual": self.actual,
            "message": self.message,
        }

    def format_console(self) -> str:
        """Return a single-line rendering of the alert.

        Floats are rounded to four decimals so a threshold like ``1500`` does not
        print as ``1780.8900000000053`` in an on-call channel.
        """
        def render(value: Any) -> str:
            """Render a value for console display.

            Args:
                value: Value from an alert threshold or observation.

            Returns:
                A short string with no trailing zeros on floats.
            """
            if isinstance(value, float):
                return "{:.4f}".format(value).rstrip("0").rstrip(".")
            return str(value)

        return "[ALERT] {} {} {} (actual: {}){}".format(
            self.metric,
            self.operator,
            render(self.threshold),
            render(self.actual),
            " - {}".format(self.message) if self.message else "",
        )


class Alerter:
    """Abstract base for alert notification backends."""

    def send(self, alerts: List[Alert]) -> None:
        """Dispatch a list of fired alerts.

        Args:
            alerts: Alerts to send. Implementations should handle empty lists
                as a no-op.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement send()")


class ConsoleAlerter(Alerter):
    """Print fired alerts to stderr."""

    def __init__(self, stream: Any = None) -> None:

        """Create a console alerter.

Args:
    stream: Writable stream to print to. Defaults to ``sys.stderr``.
            stream: Writable stream to print to. Defaults to ``sys.stderr``.
        """
        self.stream = stream if stream is not None else sys.stderr

    def send(self, alerts: List[Alert]) -> None:
        """Print each alert as a formatted line."""
        for alert in alerts:
            print(alert.format_console(), file=self.stream)


class WebhookAlerter(Alerter):
    """POST alerts to a webhook URL with exponential backoff.

Args:
    url: Endpoint receiving the JSON payload. When it is

    The webhook URL is read from an environment variable named by
    ``url_env`` in the config's ``webhook`` section. If the variable is unset
    or empty, :meth:`send` is a documented no-op (skipped with a note).
    """

    def __init__(
        self,
        url_env: str,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout_seconds: float = 10.0,
        urlopen: Optional[Callable[..., Any]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:

        """Create a webhook alerter.

Args:
    url: Endpoint receiving the JSON payload.
    max_retries: Attempts before giving up.
    backoff_seconds: Base delay for exponential backoff.
    timeout_seconds: Per-request timeout.
    sleep: Injectable sleep function; defaults to :func:`time.sleep`.
            url_env: Name of the environment variable holding the webhook URL.
            max_retries: Total delivery attempts before giving up.
            backoff_seconds: Base delay for exponential backoff between retries
                (delay doubles each retry: 1s, 2s, 4s, ...).
            timeout_seconds: Per-attempt socket timeout.
            urlopen: Injectable URL-opener for testing; defaults to
                :func:`urllib.request.urlopen`.
            sleep: Injectable sleep function; defaults to :func:`time.sleep`.
        """
        self.url_env = url_env
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    def _resolve_url(self) -> Optional[str]:
        """Return the webhook URL from the configured env var, or ``None``."""
        return os.environ.get(self.url_env) or None

    def build_payload(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Build the JSON payload sent to the webhook endpoint."""
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def send(self, alerts: List[Alert]) -> None:
        """POST the alerts, retrying transient failures.

        Delivery policy:

        * If the URL env var is missing/empty, skip with a note (no error).
        * Try up to ``max_retries`` times.
        * Retry on ``URLError`` (connection issues, timeouts).
        * HTTP error responses (4xx/5xx) are treated as terminal failures.
        * Backoff delay doubles after each failed attempt.

        Args:
            alerts: Fired alerts to deliver.

        Raises:
            RuntimeError: If all attempts fail and the webhook URL was set.
        """
        if not alerts:
            return
        url = self._resolve_url()
        if url is None:
            print(
                "[drift-sentinel] Webhook skipped: environment variable "
                "{} is not set".format(self.url_env),
                file=sys.stderr,
            )
            return
        payload = json.dumps(self.build_payload(alerts)).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._urlopen(request, timeout=self.timeout_seconds):
                    pass
                return
            except urllib.error.HTTPError as exc:
                # HTTP-level failures are terminal: the endpoint answered.
                raise RuntimeError(
                    "Webhook delivery failed: HTTP {} from {}".format(exc.code, url)
                ) from exc
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(
            "Webhook delivery failed after {} attempts: {}".format(
                self.max_retries, last_error
            )
        )


def build_alerters(webhook_config: Optional[Dict[str, Any]]) -> List[Alerter]:
    """Build the alerter list described by a webhook configuration.

    A console alerter is always included. A webhook alerter is added when
    ``webhook_config`` provides a ``url_env`` entry.

    Args:
        webhook_config: The ``webhook`` section of the config (may be ``None``).

    Returns:
        List of configured alerters.
    """
    alerters: List[Alerter] = [ConsoleAlerter()]
    if webhook_config and webhook_config.get("url_env"):
        alerters.append(
            WebhookAlerter(
                url_env=str(webhook_config["url_env"]),
                max_retries=int(webhook_config.get("max_retries", 3)),
            )
        )
    return alerters


def evaluate_alert_rules(
    metrics: Dict[str, Any],
    alert_configs: List[Dict[str, Any]],
) -> List[Alert]:
    """Evaluate every rule against a metrics dictionary.

    Args:
        metrics: Nested metrics dictionary produced by drift/LLM modules.
        alert_configs: Raw alert rule dictionaries from configuration.

    Returns:
        The list of alerts that fired (may be empty).

    Raises:
        ValueError: If any rule references an unsupported operator.
    """
    rules = [AlertRule.from_dict(raw) for raw in alert_configs]
    fired: List[Alert] = []
    for rule in rules:
        alert = rule.check(metrics)
        if alert is not None:
            fired.append(alert)
    return fired
