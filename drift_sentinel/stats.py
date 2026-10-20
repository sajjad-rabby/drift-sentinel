"""Statistical primitives.

Everything here is pure Python: no numpy, no scipy. Sorting uses a deterministic
``random``-free deterministic sorting.
- Population Stability Index (PSI) with quantile binning and epsilon smoothing.
- Kolmogorov-Smirnov two-sample statistic with an asymptotic p-value.
- Jensen-Shannon distance (base-2, bounded in [0, 1]).

No third-party dependencies: all math is implemented with ``math`` and
``random``-free deterministic sorting.
"""

import math
from typing import Dict, List, Sequence, Tuple

EPSILON = 1e-6

# Severity bands for PSI: (upper bound exclusive, label).
PSI_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.1, "stable"),
    (0.25, "moderate"),
    (float("inf"), "significant"),
)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sample.

Uses the same ``(n - 1) * q`` interpolation as numpy's default method.

    ``q`` must be in [0, 100]. Linear interpolation between neighbouring
    order statistics (numpy's default method).
    """
    if len(sorted_values) == 0:
        raise ValueError("percentile() requires a non-empty sequence")
    if q < 0 or q > 100:
        raise ValueError("q must be within [0, 100]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def quantile_bin_edges(reference: Sequence[float], bins: int) -> List[float]:
    """Compute quantile bin edges from a reference sample.

Args:
    reference: Values defining the edges.
    bins: Number of bins.

Duplicate interior edges are collapsed so every value falls into exactly one bin.

    Returns ``bins - 1`` interior cut points. Consumers add +/- infinity at
    the ends so every value falls into exactly one bin.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    ordered = sorted(float(v) for v in reference)
    if len(ordered) < 2:
        raise ValueError("reference must contain at least 2 values")
    edges: List[float] = []
    for i in range(1, bins):
        edges.append(percentile(ordered, 100.0 * i / bins))
    return edges


def histogram(values: Sequence[float], edges: Sequence[float]) -> List[int]:
    """Count values into bins delimited by interior edges ([-inf, e1), ...)."""
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, edge in enumerate(edges):
            if v < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

Bins are cut from the reference's own quantiles, so the statistic is
directional: swapping the arguments re-bins against the other distribution.
Both histograms are smoothed with ``EPSILON`` before proportions are taken.
PSI = sum((cur - ref) * ln(cur / ref)) over bins.

Returns:
    The PSI value, and ``None`` when the sample is too small to bin.

    Bin edges are the reference quantiles; both histograms are smoothed with
    ``EPSILON`` before proportions are taken. PSI = sum((cur - ref) * ln(cur / ref)).
    """
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("psi() requires non-empty samples")
    edges = quantile_bin_edges(reference, bins)
    ref_counts = histogram(reference, edges)
    cur_counts = histogram(current, edges)
    total = 0.0
    for rc, cc in zip(ref_counts, cur_counts):
        ref_pct = rc / len(reference)
        cur_pct = cc / len(current)
        ref_pct = max(ref_pct, EPSILON)
        cur_pct = max(cur_pct, EPSILON)
        total += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
    return total


def _ecdf(sorted_values: Sequence[float], x: float) -> float:
    """Empirical CDF of a sorted sample evaluated at x."""
    count = 0
    for v in sorted_values:
        if v <= x:
            count += 1
        else:
            break
    return count / len(sorted_values)


def ks_statistic(sample1: Sequence[float], sample2: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: sup |F1(x) - F2(x)|."""
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("ks_statistic() requires non-empty samples")
    s1 = sorted(float(v) for v in sample1)
    s2 = sorted(float(v) for v in sample2)
    points = set(s1) | set(s2)
    best = 0.0
    for x in points:
        diff = abs(_ecdf(s1, x) - _ecdf(s2, x))
        if diff > best:
            best = diff
    return best


def ks_pvalue(d: float, n: int, m: int) -> float:
    """Asymptotic two-sample Kolmogorov-Smirnov p-value.

Applies the Stephens small-sample correction, then returns
``p = 2 * sum((-1)^(k-1) * exp(-2 * k^2 * lam^2))`` over the series.

This approximation is not reliable on very small samples.

    ``d`` is the KS statistic; ``n`` and ``m`` are the two sample sizes.
    Uses the standard exponential approximation
    ``lam = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * d`` with ``ne = n*m/(n+m)``,
    then returns ``p = 2 * sum((-1)^(k-1) * exp(-2 * k^2 * lam^2))`` for k >= 1.
    """
    if n <= 0 or m <= 0:
        raise ValueError("sample sizes must be positive")
    ne = (n * m) / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam <= 0.0:
        return 1.0
    total = 0.0
    k = 1
    while True:
        term = (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
        k += 1
        if k > 100:
            break
    p = 2.0 * total
    return max(0.0, min(1.0, p))


def _discrete_distribution(values: Sequence[str]) -> Dict[str, float]:
    """Proportions of each category in a sample."""
    if len(values) == 0:
        raise ValueError("distribution requires a non-empty sample")
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = len(values)
    return {key: count / n for key, count in counts.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(P || Q) with epsilon smoothing on zero-mass categories (base e)."""
    total = 0.0
    for key, pv in p.items():
        qv = max(q.get(key, 0.0), EPSILON)
        pv = max(pv, EPSILON)
        total += pv * math.log(pv / qv)
    return total


def js_distance(sample1: Sequence[str], sample2: Sequence[str]) -> float:
    """Jensen-Shannon distance between two categorical samples.

Returns:
    A value in ``[0, 1]``, symmetric and finite even for disjoint support. Computed as the
square root of the divergence between sample and
midpoint distribution; the distance is sqrt(JS) and lies in [0, 1].

    JS divergence (base 2) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) with M the
    midpoint distribution; the distance is sqrt(JS) and lies in [0, 1].
    """
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("js_distance() requires non-empty samples")
    p = _discrete_distribution(sample1)
    q = _discrete_distribution(sample2)
    m: Dict[str, float] = {}
    for key in set(p) | set(q):
        m[key] = 0.5 * p.get(key, 0.0) + 0.5 * q.get(key, 0.0)
    divergence = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    # Base-2 conversion then distance.
    js_bits = max(0.0, min(1.0, divergence / math.log(2.0)))
    return math.sqrt(js_bits)


def psi_band(psi_value: float) -> str:
    """Map a PSI value to its severity band label."""
    for upper, label in PSI_BANDS:
        if psi_value < upper:
            return label
    return "significant"
