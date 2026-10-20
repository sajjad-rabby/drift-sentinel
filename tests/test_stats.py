"""Tests for the statistical primitives."""

from __future__ import annotations

import math

import pytest

from drift_sentinel.stats import (
    EPSILON,
    PSI_BANDS,
    histogram,
    js_distance,
    ks_pvalue,
    ks_statistic,
    percentile,
    psi,
    psi_band,
    quantile_bin_edges,
)


class TestPercentile:
    def test_endpoints(self):
        assert percentile([1, 2, 3, 4], 0) == 1
        assert percentile([1, 2, 3, 4], 100) == 4

    def test_median_of_odd_length(self):
        assert percentile([1, 2, 3], 50) == 2

    def test_interpolates_linearly(self):
        # rank = (4-1) * 0.5 = 1.5 -> midway between 2 and 3
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_single_value(self):
        assert percentile([7], 50) == 7

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            percentile([], 50)

    @pytest.mark.parametrize("q", [-1, 101])
    def test_rejects_out_of_range_q(self, q):
        with pytest.raises(ValueError):
            percentile([1, 2, 3], q)

    def test_empty_percentile_bounds_are_accepted(self):
        assert percentile([1, 2, 3], 0) == 1
        assert percentile([1, 2, 3], 100) == 3


class TestQuantileBins:
    def test_returns_bins_minus_one_edges(self):
        edges = quantile_bin_edges([float(i) for i in range(100)], bins=10)
        assert len(edges) == 9

    def test_edges_are_sorted(self):
        edges = quantile_bin_edges([float(i) for i in range(100)], bins=5)
        assert edges == sorted(edges)

    def test_unsorted_input_is_handled(self):
        edges = quantile_bin_edges([5, 1, 4, 2, 3], bins=5)
        assert edges == sorted(edges)

    def test_rejects_single_value_reference(self):
        with pytest.raises(ValueError):
            quantile_bin_edges([1.0], bins=5)

    @pytest.mark.parametrize("bins", [0, 1])
    def test_rejects_too_few_bins(self, bins):
        with pytest.raises(ValueError):
            quantile_bin_edges([1.0, 2.0, 3.0], bins=bins)


class TestHistogram:
    def test_counts_all_values(self):
        counts = histogram([1.0, 2.0, 3.0, 4.0], [2.5, 3.5])
        assert sum(counts) == 4

    def test_bins_are_left_closed(self):
        # value exactly on an edge falls into the upper bin
        counts = histogram([2.5], [2.5])
        assert counts == [0, 1]

    def test_no_edges_gives_single_bin(self):
        assert histogram([1.0, 2.0], []) == [2]

    def test_values_below_first_edge(self):
        assert histogram([-100.0], [0.0, 10.0]) == [1, 0, 0]

    def test_values_above_last_edge(self):
        assert histogram([100.0], [0.0, 10.0]) == [0, 0, 1]

    def test_conserves_mass(self, normal_reference):
        edges = quantile_bin_edges(normal_reference, bins=10)
        assert sum(histogram(normal_reference, edges)) == len(normal_reference)


class TestPsi:
    def test_identical_distributions_score_near_zero(self, normal_reference):
        assert psi(normal_reference, list(normal_reference)) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_scores_higher(self, normal_reference, shifted_current):
        assert psi(normal_reference, shifted_current) > 0.1

    def test_is_not_symmetric_by_design(self, normal_reference, shifted_current):
        """PSI is directional: bin edges come from the *reference* sample.

        Swapping the arguments therefore re-bins against the other
        distribution, so the two values are close but not identical. This is
        why JS distance and KS are the right tools when you need a symmetric
        comparison.
        """
        forward = psi(normal_reference, shifted_current)
        backward = psi(shifted_current, normal_reference)
        assert forward > 0 and backward > 0
        assert forward != backward

    def test_is_non_negative(self, normal_reference, shifted_current):
        assert psi(normal_reference, shifted_current) >= 0

    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError):
            psi([], [1.0])
        with pytest.raises(ValueError):
            psi([1.0], [])

    def test_empty_bins_are_smoothed_not_infinite(self, normal_reference):
        """A current sample with a mass the reference never had must stay finite."""
        reference = [float(i) for i in range(200)]
        current = reference + [1000.0] * 50
        value = psi(reference, current, bins=10)
        assert math.isfinite(value)
        assert value > 0

    def test_more_bins_is_at_least_as_sensitive(self, normal_reference, shifted_current):
        coarse = psi(normal_reference, shifted_current, bins=5)
        fine = psi(normal_reference, shifted_current, bins=20)
        assert fine > coarse


class TestPsiBand:
    def test_band_boundaries(self):
        assert psi_band(0.0) == "stable"
        assert psi_band(0.099) == "stable"
        assert psi_band(0.1) == "moderate"
        assert psi_band(0.24) == "moderate"
        assert psi_band(0.25) == "significant"
        assert psi_band(5.0) == "significant"

    def test_bands_cover_the_range(self):
        assert PSI_BANDS[-1][0] == float("inf")


class TestKsStatistic:
    def test_identical_samples(self, normal_reference):
        assert ks_statistic(normal_reference, list(normal_reference)) == pytest.approx(0.0)

    def test_disjoint_samples_gives_one(self):
        a = [0.0, 0.1, 0.2]
        b = [10.0, 11.0, 12.0]
        assert ks_statistic(a, b) == pytest.approx(1.0)

    def test_bounded_by_one(self, normal_reference, shifted_current):
        assert 0.0 <= ks_statistic(normal_reference, shifted_current) <= 1.0

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            ks_statistic([], [1.0])

    def test_is_symmetric(self, normal_reference, shifted_current):
        assert ks_statistic(normal_reference, shifted_current) == pytest.approx(
            ks_statistic(shifted_current, normal_reference)
        )


class TestKsPvalue:
    def test_large_statistic_gives_small_pvalue(self):
        assert ks_pvalue(0.9, 100, 100) < 0.001

    def test_zero_statistic_gives_one(self):
        assert ks_pvalue(0.0, 100, 100) == 1.0

    def test_is_bounded(self, normal_reference, shifted_current):
        d = ks_statistic(normal_reference, shifted_current)
        p = ks_pvalue(d, len(normal_reference), len(shifted_current))
        assert 0.0 <= p <= 1.0

    def test_decreases_with_statistic(self):
        assert ks_pvalue(0.2, 100, 100) > ks_pvalue(0.5, 100, 100)

    def test_larger_samples_give_smaller_pvalues(self):
        """The asymptotic approximation needs a large enough n to be useful.

        At n=3 it can still return p < 0.05, so it must not be trusted on tiny
        samples; the trend across sample sizes is the property worth asserting.
        """
        assert ks_pvalue(0.5, 20, 20) > ks_pvalue(0.5, 200, 200)
        assert ks_pvalue(0.5, 200, 200) > ks_pvalue(0.5, 5000, 5000)

    @pytest.mark.parametrize("n,m", [(0, 10), (10, 0), (-1, 5)])
    def test_rejects_bad_sample_sizes(self, n, m):
        with pytest.raises(ValueError):
            ks_pvalue(0.3, n, m)


class TestJsDistance:
    def test_identical_samples(self, categorical_reference):
        assert js_distance(categorical_reference, list(categorical_reference)) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_categories_gives_one(self):
        assert js_distance(["a"] * 50, ["b"] * 50) == pytest.approx(1.0)

    def test_bounded_in_unit_interval(self):
        assert 0.0 <= js_distance(["a", "b", "c"], ["x", "y"]) <= 1.0

    def test_is_symmetric(self):
        assert js_distance(["a", "b", "a"], ["b", "c", "c"]) == pytest.approx(
            js_distance(["b", "c", "c"], ["a", "b", "a"])
        )

    def test_similarity_ranks_correctly(self):
        base = ["a"] * 60 + ["b"] * 40
        close = ["a"] * 55 + ["b"] * 45
        far = ["a"] * 10 + ["b"] * 90
        assert js_distance(base, close) < js_distance(base, far)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            js_distance([], ["a"])

    def test_disjoint_support_is_finite(self):
        """KL over disjoint supports is infinite before smoothing."""
        value = js_distance(["a"] * 10, ["b"] * 10)
        assert math.isfinite(value)


class TestEpsilon:
    def test_positive_and_small(self):
        assert 0 < EPSILON < 1e-5
