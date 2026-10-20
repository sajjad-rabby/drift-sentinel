"""LLM behaviour metrics.

Each JSONL record describes one call:
- ``model``: model identifier string.

- ``prompt``: the user prompt text.
- ``response``: the model's response text.
- ``latency_ms``: round-trip latency in milliseconds (number).
- ``prompt_tokens``: number of prompt tokens (number).
- ``completion_tokens``: number of completion tokens (number).
- ``model``: model identifier string.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .stats import percentile


@dataclass
class PercentileStats:
    """Summary statistics over a numeric sample."""

    count: int
    mean: float
    minimum: float
    p50: float
    p90: float
    p99: float
    maximum: float

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serializable dict of the stats."""
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.minimum,
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "max": self.maximum,
        }


@dataclass
class LLMMetrics:
    """Aggregated metrics for a batch of LLM log entries."""

    entry_count: int = 0
    response_length: Optional[PercentileStats] = None
    latency_ms: Optional[PercentileStats] = None
    refusal_rate: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    jaccard_similarity: Optional[float] = None
    per_model: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all metrics."""
        out: Dict[str, Any] = {"entry_count": self.entry_count}
        if self.response_length is not None:
            out["response_length"] = self.response_length.to_dict()
        if self.latency_ms is not None:
            out["latency_ms"] = self.latency_ms.to_dict()
        out["refusal_rate"] = self.refusal_rate
        out["total_prompt_tokens"] = self.total_prompt_tokens
        out["total_completion_tokens"] = self.total_completion_tokens
        out["total_cost"] = self.total_cost
        if self.jaccard_similarity is not None:
            out["jaccard_similarity"] = self.jaccard_similarity
        if self.per_model:
            out["per_model"] = self.per_model
        return out


# Lowercase phrases that signal a model refusal.
REFUSAL_PHRASES: Tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
)

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _numeric_series(entries: Sequence[Dict[str, Any]], key: str) -> List[float]:
    """Collect non-null numeric values for ``key`` across entries."""
    series: List[float] = []
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            series.append(float(value))
    return series


def _sum_stats(values: List[float]) -> Optional[PercentileStats]:
    """Build PercentileStats for a sample, or None when empty."""
    if not values:
        return None
    mean = sum(values) / len(values)
    return PercentileStats(
        count=len(values),
        mean=mean,
        minimum=min(values),
        p50=percentile(values, 50.0),
        p90=percentile(values, 90.0),
        p99=percentile(values, 99.0),
        maximum=max(values),
    )


def _word_set(text: str) -> set:
    """Extract a lowercase word set from text (split on non-alphanumerics)."""
    return {token for token in _SPLIT_RE.split(text.lower()) if token}


def jaccard_similarity(responses: Sequence[str], reference: str) -> float:
    """Mean Jaccard similarity of responses against a reference.

Each response is tokenised into a word set; similarity is
intersection over union, averaged across responses.

Raises:
    ValueError: If no reference is supplied, because then "maximum" degenerates to the single comparison per respon

    Each response is compared against the reference by word-set Jaccard
    similarity; the returned value is the mean of the maximum similarity each
    response achieved (each response is compared against the reference set,
    so "maximum" degenerates to the single comparison per response).
    """
    ref_set = _word_set(reference)
    if not ref_set:
        return 0.0
    scores: List[float] = []
    for response in responses:
        resp_set = _word_set(response)
        if not resp_set:
            scores.append(0.0)
            continue
        intersection = len(resp_set & ref_set)
        union = len(resp_set | ref_set)
        scores.append(intersection / union if union else 0.0)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def is_refusal(text: str) -> bool:
    """Return True when ``text`` contains a known refusal phrase."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def estimate_cost(
    total_prompt_tokens: int,
    total_completion_tokens: int,
    model: str,
    price_table: Dict[str, Dict[str, float]],
) -> float:
    """Estimate total cost from token counts and a price table.

Args:
    records: Parsed call log entries.
    price_table: Mapping of model name to ``prompt_per_1k`` and

    The price table maps model name to a dict with ``prompt_per_1k`` and
    ``completion_per_1k`` keys. Unknown models cost 0.0.
    """
    prices = price_table.get(model)
    if not prices:
        return 0.0
    prompt_cost = total_prompt_tokens / 1000.0 * float(prices.get("prompt_per_1k", 0.0))
    completion_cost = total_completion_tokens / 1000.0 * float(
        prices.get("completion_per_1k", 0.0)
    )
    return prompt_cost + completion_cost


def compute_llm_metrics(
    entries: Sequence[Dict[str, Any]],
    reference_response: Optional[str] = None,
    price_table: Optional[Dict[str, Dict[str, float]]] = None,
) -> LLMMetrics:
    """Compute latency, refusal, token, cost and similarity metrics.

    Args:
        entries: Sequence of log-entry dicts (see module docstring).
        reference_response: Optional reference text for Jaccard similarity.
        price_table: Optional model price table for cost estimation.

    Returns:
        An :class:`LLMMetrics` instance.
    """
    metrics = LLMMetrics()
    metrics.entry_count = len(entries)

    length_series = _numeric_series(entries, "response_chars")
    if not length_series:
        # Fall back to measuring response text length directly.
        lengths: List[float] = []
        for entry in entries:
            response = entry.get("response")
            if isinstance(response, str):
                lengths.append(float(len(response)))
        length_series = lengths
    metrics.response_length = _sum_stats(length_series)
    metrics.latency_ms = _sum_stats(_numeric_series(entries, "latency_ms"))

    refusals = 0
    responses = 0
    for entry in entries:
        response = entry.get("response")
        if isinstance(response, str):
            responses += 1
            if is_refusal(response):
                refusals += 1
    if responses:
        metrics.refusal_rate = refusals / responses

    model_tokens: Dict[str, Dict[str, int]] = {}
    for entry in entries:
        prompt_tokens = entry.get("prompt_tokens")
        completion_tokens = entry.get("completion_tokens")
        if isinstance(prompt_tokens, (int, float)) and not isinstance(prompt_tokens, bool):
            prompt_tokens = int(prompt_tokens)
        else:
            prompt_tokens = 0
        if isinstance(completion_tokens, (int, float)) and not isinstance(
            completion_tokens, bool
        ):
            completion_tokens = int(completion_tokens)
        else:
            completion_tokens = 0
        metrics.total_prompt_tokens += prompt_tokens
        metrics.total_completion_tokens += completion_tokens
        model = entry.get("model")
        if isinstance(model, str) and model:
            bucket = model_tokens.setdefault(
                model, {"prompt_tokens": 0, "completion_tokens": 0}
            )
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens

    table = price_table or {}
    if model_tokens:
        for model, bucket in model_tokens.items():
            cost = estimate_cost(
                bucket["prompt_tokens"],
                bucket["completion_tokens"],
                model,
                table,
            )
            metrics.total_cost += cost
            metrics.per_model[model] = {
                "prompt_tokens": float(bucket["prompt_tokens"]),
                "completion_tokens": float(bucket["completion_tokens"]),
                "estimated_cost_usd": round(cost, 6),
            }
    else:
        model = None
        for entry in entries:
            value = entry.get("model")
            if isinstance(value, str) and value:
                model = value
                break
        if model is not None:
            cost = estimate_cost(
                metrics.total_prompt_tokens,
                metrics.total_completion_tokens,
                model,
                table,
            )
            metrics.total_cost += cost

    if reference_response:
        texts = [
            entry["response"]
            for entry in entries
            if isinstance(entry.get("response"), str)
        ]
        if texts:
            metrics.jaccard_similarity = jaccard_similarity(texts, reference_response)

    return metrics
