"""Dataset-level drift detection over CSV-like column data."""

from typing import Any, Dict, List, Optional, Sequence

from . import stats

NUMERIC_CATEGORICAL = "categorical"
NUMERIC_NUMERIC = "numeric"


class FeatureDrift(object):
    """Result of comparing one feature across reference and current data."""

    def __init__(
        self,
        feature: str,
        kind: str,
        metric: str,
        value: float,
        pvalue: Optional[float],
        band: str,
    ) -> None:
        self.feature = feature
        self.kind = kind
        self.metric = metric
        self.value = value
        self.pvalue = pvalue
        self.band = band

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible mapping.

        Returns:
            JSON-compatible mapping, using ``None`` for an absent p-value.
        """
        return {
            "feature": self.feature,
            "kind": self.kind,
            "metric": self.metric,
            "value": round(self.value, 6),
            "pvalue": None if self.pvalue is None else round(self.pvalue, 6),
            "band": self.band,
        }


class DriftReport(object):
    """Aggregated drift results for a dataset comparison."""

    def __init__(self, features: List[FeatureDrift], verdict: str) -> None:
        self.features = features
        self.verdict = verdict

    def to_dict(self) -> Dict[str, Any]:
        """Return a mapping with verdict and per-feature features.

        Returns:
            Mapping with ``verdict`` and the per-feature ``features`` list.
        """
        return {
            "verdict": self.verdict,
            "features": [f.to_dict() for f in self.features],
        }


def _is_numeric(values: Sequence[Any]) -> bool:
    """True when every non-empty value parses as a float."""
    non_empty = [v for v in values if v is not None and str(v).strip() != ""]
    if not non_empty:
        return False
    for v in non_empty:
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def _to_floats(values: Sequence[Any]) -> List[float]:
    return [float(v) for v in values if v is not None and str(v).strip() != ""]


def _to_strings(values: Sequence[Any]) -> List[str]:
    return [str(v) for v in values if v is not None and str(v).strip() != ""]


def compare_feature(
    name: str,
    reference: Sequence[Any],
    current: Sequence[Any],
    bins: int = 10,
    psi_bands: Optional[Dict[str, float]] = None,
) -> Optional[FeatureDrift]:
    """Compare one feature across two samples.

Numeric features are scored with PSI; categorical features with
Jensen-Shannon distance. A numeric reference with fewer than two distinct values cannot support a
distribution estimate and is compared categorically instead.

Returns:

    Numeric features use PSI (with band mapping); categorical features use
    Jensen-Shannon distance with the same fixed banding (stable < 0.1,
    moderate < 0.25, significant otherwise). ``psi_bands`` overrides the
    numeric band thresholds, e.g. {"stable": 0.1, "moderate": 0.25}.

    Type detection requires *both* sides to be numeric. A column that is
    numeric in the reference but picks up a non-numeric value in production is
    therefore compared categorically rather than crashing - that is itself a
    signal worth seeing rather than an error.

    Returns None when a comparison is not meaningful: either side is empty, or
    a numeric column has fewer than two values (PSI needs a reference
    distribution).
    """
    ref_nums = _is_numeric(reference)
    cur_nums = _is_numeric(current)
    if ref_nums and cur_nums:
        r = _to_floats(reference)
        c = _to_floats(current)
        if len(r) < 2 or len(c) < 2:
            return None
        value = stats.psi(r, c, bins=bins)
        band = _band_for_psi(value, psi_bands)
        return FeatureDrift(name, NUMERIC_NUMERIC, "psi", value, None, band)
    ref_str = _to_strings(reference)
    cur_str = _to_strings(current)
    if not ref_str or not cur_str:
        return None
    value = stats.js_distance(ref_str, cur_str)
    band = _band_for_js(value)
    return FeatureDrift(name, NUMERIC_CATEGORICAL, "js_distance", value, None, band)


def _band_for_psi(value: float, psi_bands: Optional[Dict[str, float]]) -> str:
    if psi_bands:
        stable = psi_bands.get("stable", 0.1)
        moderate = psi_bands.get("moderate", 0.25)
    else:
        stable, moderate = 0.1, 0.25
    if value < stable:
        return "stable"
    if value < moderate:
        return "moderate"
    return "significant"


def _band_for_js(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def compare_datasets(
    reference: Dict[str, Sequence[Any]],
    current: Dict[str, Sequence[Any]],
    bins: int = 10,
    psi_bands: Optional[Dict[str, float]] = None,
) -> DriftReport:
    """Compare all shared columns and derive an aggregate verdict."""
    features: List[FeatureDrift] = []
    for name in sorted(set(reference) & set(current)):
        result = compare_feature(name, reference[name], current[name], bins=bins, psi_bands=psi_bands)
        if result is not None:
            features.append(result)
    if not features:
        verdict = "no_data"
    elif any(f.band == "significant" for f in features):
        verdict = "significant_drift"
    elif any(f.band == "moderate" for f in features):
        verdict = "moderate_drift"
    else:
        verdict = "stable"
    return DriftReport(features, verdict)
