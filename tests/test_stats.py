"""Unit tests for mveac.evaluation.stats: Cohen's d, Wilcoxon p-values, and the
Holm-Bonferroni / Benjamini-Hochberg multiplicity corrections."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mveac.evaluation.stats import (
    benjamini_hochberg,
    cohens_d,
    compare,
    holm_bonferroni,
    is_practically_meaningful,
    wilcoxon_p,
)


class TestCohensD:
    def test_zero_mean_difference_is_zero(self):
        diff = np.array([-1.0, 1.0, -2.0, 2.0])
        assert cohens_d(diff) == pytest.approx(0.0, abs=1e-9)

    def test_matches_hand_computation(self):
        diff = np.array([1.0, 2.0, 3.0])  # mean=2, sample sd=1
        assert cohens_d(diff) == pytest.approx(2.0)

    def test_zero_variance_returns_zero_not_infinity(self):
        diff = np.array([5.0, 5.0, 5.0])
        assert cohens_d(diff) == 0.0

    def test_ignores_non_finite_values(self):
        diff = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
        assert cohens_d(diff) == pytest.approx(2.0)

    def test_too_few_observations_is_nan(self):
        assert np.isnan(cohens_d(np.array([1.0])))


class TestWilcoxonP:
    def test_all_zero_differences_is_not_significant(self):
        assert wilcoxon_p(np.zeros(10)) == 1.0

    def test_clear_shift_is_significant(self):
        rng = np.random.default_rng(0)
        diff = rng.normal(loc=5.0, scale=1.0, size=200)  # far from zero
        assert wilcoxon_p(diff) < 0.001


class TestMultiplicityCorrection:
    def test_holm_is_at_least_as_conservative_as_uncorrected(self):
        p = np.array([0.001, 0.01, 0.04, 0.20, 0.80])
        rejected = holm_bonferroni(p, alpha=0.05)
        assert rejected.sum() <= (p < 0.05).sum()

    def test_holm_rejects_the_smallest_p_values_first(self):
        p = np.array([0.001, 0.5, 0.9])
        rejected = holm_bonferroni(p, alpha=0.05)
        assert rejected[0]  # 0.001 < 0.05/3 always rejects
        assert not rejected[1] and not rejected[2]

    def test_benjamini_hochberg_is_at_least_as_powerful_as_holm(self):
        p = np.array([0.001, 0.01, 0.02, 0.03, 0.9])
        holm = holm_bonferroni(p, alpha=0.05)
        bh = benjamini_hochberg(p, alpha=0.05)
        assert bh.sum() >= holm.sum()

    def test_all_large_p_values_reject_nothing(self):
        p = np.array([0.5, 0.6, 0.7])
        assert not holm_bonferroni(p).any()
        assert not benjamini_hochberg(p).any()


class TestCompare:
    def test_matching_series_have_zero_delta(self):
        values = pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30])
        users = pd.Series(["u1", "u2", "u3"], index=[10, 20, 30])
        result = compare(values, values, users)
        assert result["delta"] == pytest.approx(0.0)

    def test_minimize_flag_flips_the_sign_convention(self):
        # b is uniformly LOWER (better, for a minimize metric like ERR/TCI) than a.
        a = pd.Series([2.0, 3.0, 4.0], index=[1, 2, 3])
        b = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
        users = pd.Series(["u1", "u2", "u3"], index=[1, 2, 3])
        maximize_result = compare(a, b, users, minimize=False)
        minimize_result = compare(a, b, users, minimize=True)
        # Same underlying data, opposite sign under the two conventions.
        assert maximize_result["d_impression"] == pytest.approx(-minimize_result["d_impression"])

    def test_user_clustering_reduces_effective_sample_size(self):
        # Three impressions from the same two users -> only 2 user-level observations.
        a = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
        b = pd.Series([0.0, 0.0, 0.0], index=[1, 2, 3])
        users = pd.Series(["u1", "u1", "u2"], index=[1, 2, 3])
        result = compare(a, b, users)
        assert result["n_impressions"] == 3
        assert result["n_users"] == 2


class TestPracticalSignificanceThreshold:
    def test_matches_the_papers_010_threshold(self):
        assert is_practically_meaningful(0.11)
        assert not is_practically_meaningful(0.09)
        assert is_practically_meaningful(-0.50)
