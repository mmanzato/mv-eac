"""Unit tests for the calibration primitives and the EAC/MV-EAC rerankers,
on small synthetic candidate sets with hand-verifiable behavior. No dataset
or pipeline output is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from mveac.calibration.eac import multi_view_eac, single_view_eac, traditional_rerank
from mveac.calibration.traditional import kl_divergence, minmax_normalize

VOCAB = ["a", "b"]
ARTICLE_MAP = {1: ["a"], 2: ["b"], 3: ["a"]}
CANDIDATES = [(1, 1.0), (2, 0.9), (3, 0.5)]  # article 1 is most "relevant" per the base recommender


class TestKLDivergence:
    def test_identical_distributions_have_zero_divergence(self):
        p = np.array([0.5, 0.5])
        assert kl_divergence(p, p, epsilon=1e-10) == pytest.approx(0.0, abs=1e-6)

    def test_diverges_sharply_when_q_has_no_mass_where_p_does(self):
        p, q = np.array([0.0, 1.0]), np.array([1.0, 0.0])
        # Q assigns (essentially) zero probability to the one value P puts all its
        # mass on -- the epsilon smoothing keeps this finite but large.
        assert kl_divergence(p, q, epsilon=1e-10) > 10.0

    def test_symmetric_case_matches_hand_computation(self):
        p, q = np.array([0.0, 1.0]), np.array([0.5, 0.5])
        assert kl_divergence(p, q, epsilon=1e-10) == pytest.approx(np.log(2), abs=1e-4)


class TestMinMaxNormalize:
    def test_spreads_values_into_unit_interval(self):
        assert minmax_normalize([1.0, 2.0, 3.0]) == pytest.approx([0.0, 0.5, 1.0])

    def test_constant_list_maps_to_all_ones(self):
        # A tie in relevance should not be treated as "everything is worst" --
        # the convention here is that a flat input is scored as uniformly best.
        assert minmax_normalize([5.0, 5.0, 5.0]) == pytest.approx([1.0, 1.0, 1.0])


class TestTraditionalRerank:
    def test_zero_lambda_reduces_to_sorting_by_relevance(self):
        # With lambda=0 the score is pure (min-max normalized) relevance, so the
        # output order must exactly match the base recommender's own ranking.
        profile = np.array([0.0, 1.0])  # irrelevant when lambda=0
        result = traditional_rerank(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.0, k=3)
        assert [aid for aid, _ in result] == [1, 2, 3]

    def test_full_lambda_prioritizes_calibration_over_relevance(self):
        # A user profile concentrated entirely on "b" should pull the reranker
        # toward article 2 (the only "b"-labeled candidate) FIRST, even though
        # it is not the most relevant candidate by the base recommender's score.
        profile = np.array([0.0, 1.0])
        result = traditional_rerank(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=1.0, k=1)
        assert result[0][0] == 2

    def test_returns_at_most_k_items(self):
        profile = np.array([0.5, 0.5])
        result = traditional_rerank(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.5, k=2)
        assert len(result) == 2

    def test_never_invents_an_article_id(self):
        profile = np.array([0.5, 0.5])
        result = traditional_rerank(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.7, k=3)
        assert set(aid for aid, _ in result) <= {1, 2, 3}


class TestSingleViewEAC:
    def test_zero_beta_matches_traditional_calibration_exactly(self):
        # EAC with beta=0 has no exploration term at all -- it must be bit-for-bit
        # identical to Traditional calibration at the same lambda.
        profile = np.array([0.2, 0.8])
        traditional = traditional_rerank(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.6, k=3)
        eac = single_view_eac(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.6, beta=0.0, k=3)
        assert traditional == eac

    def test_positive_beta_changes_at_least_one_score(self):
        # Turning on exploration must change *something* relative to beta=0
        # (the UCB term is only exactly zero for every candidate in
        # degenerate cases, none of which apply here).
        profile = np.array([0.2, 0.8])
        no_exploration = single_view_eac(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.5, beta=0.0, k=3)
        with_exploration = single_view_eac(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.5, beta=0.9, k=3)
        assert no_exploration != with_exploration


class TestMultiViewEAC:
    def test_single_active_view_matches_single_view_eac(self):
        # MV-EAC with only one view carrying positive weight must reduce exactly
        # to single-view EAC on that view.
        profile = np.array([0.3, 0.7])
        single = single_view_eac(CANDIDATES, profile, ARTICLE_MAP, VOCAB, lam=0.5, beta=0.5, k=3)
        multi = multi_view_eac(
            CANDIDATES, user_profiles={"category": profile},
            article_maps={"category": ARTICLE_MAP}, vocabs={"category": VOCAB},
            weights={"category": 1.0}, lam=0.5, beta=0.5, k=3,
        )
        assert single == multi

    def test_zero_weight_view_has_no_effect(self):
        # A view included in user_profiles/article_maps/vocabs but assigned zero
        # weight must be completely ignored -- this is how the RQ2 ablation and
        # RQ4 weight sweep are implemented (by zeroing a weight, not by deleting
        # the view's data).
        profile_a = np.array([0.3, 0.7])
        profile_b_irrelevant = np.array([0.9, 0.1])  # should have zero influence
        only_a = multi_view_eac(
            CANDIDATES, user_profiles={"category": profile_a},
            article_maps={"category": ARTICLE_MAP}, vocabs={"category": VOCAB},
            weights={"category": 1.0}, lam=0.5, beta=0.5, k=3,
        )
        a_and_zeroed_b = multi_view_eac(
            CANDIDATES, user_profiles={"category": profile_a, "sentiment": profile_b_irrelevant},
            article_maps={"category": ARTICLE_MAP, "sentiment": ARTICLE_MAP}, vocabs={"category": VOCAB, "sentiment": VOCAB},
            weights={"category": 1.0, "sentiment": 0.0}, lam=0.5, beta=0.5, k=3,
        )
        assert only_a == a_and_zeroed_b
