"""Tests for the release-1.4.0 additions (second revision of the paper): the general
greedy reranker (per-view normalization, exploration components, beta = inf), the
diversification baselines, and the position-sensitive metrics. Synthetic data only."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mveac.calibration.baselines import dedup, dpp, mmr, xquad
from mveac.calibration.eac import _minmax_active, eac_rerank, multi_view_eac, single_view_eac
from mveac.evaluation.stats import cohens_d, wilcoxon_p
from mveac.metrics.position import (
    calibration_error,
    category_ild_entropy,
    position_weights,
    weighted_err,
    weighted_tci,
)

rng = np.random.default_rng(7)
VOCABS = {"category": [f"c{i}" for i in range(4)], "topic": [f"t{i}" for i in range(6)],
          "entity": [f"e{i}" for i in range(12)]}
N_ART = 30
MAPS = {
    "category": {a: [f"c{rng.integers(4)}"] for a in range(N_ART)},
    "topic": {a: [f"t{x}" for x in rng.choice(6, rng.integers(1, 3), replace=False)] for a in range(N_ART)},
    # some articles carry no entity, as in EB-NeRD
    "entity": {a: ([f"e{x}" for x in rng.choice(12, rng.integers(1, 4), replace=False)] if a % 4 else [])
               for a in range(N_ART)},
}


def _profile(n):
    p = rng.random(n) ** 3
    return p / p.sum()


def _candidates(n):
    ids = rng.choice(N_ART, n, replace=False)
    return [(int(a), float(s)) for a, s in zip(ids, np.sort(rng.random(n))[::-1])]


@pytest.mark.parametrize("seed", range(20))
def test_unnormalized_general_reranker_reproduces_published_mv_eac(seed):
    global rng
    rng = np.random.default_rng(seed)
    cands = _candidates(int(rng.integers(5, 25)))
    prof = {v: _profile(len(VOCABS[v])) for v in VOCABS}
    w = {v: 1 / 3 for v in VOCABS}
    lam, beta = float(rng.choice([0.1, 0.5, 0.9])), float(rng.choice([0.0, 0.5, 2.0, 10.0]))
    old = multi_view_eac(cands, prof, MAPS, VOCABS, w, lam, beta, 10)
    new = eac_rerank(cands, prof, MAPS, VOCABS, w, lam, beta, 10, normalize=False)
    assert [a for a, _ in old] == [a for a, _ in new]


@pytest.mark.parametrize("view", ["category", "topic", "entity"])
def test_unnormalized_general_reranker_reproduces_single_view_eac(view):
    cands = _candidates(20)
    prof = _profile(len(VOCABS[view]))
    old = single_view_eac(cands, prof, MAPS[view], VOCABS[view], 0.7, 1.5, 10)
    new = eac_rerank(cands, {view: prof}, MAPS, VOCABS, {view: 1.0}, 0.7, 1.5, 10, normalize=False)
    assert [a for a, _ in old] == [a for a, _ in new]


def test_minmax_active_ignores_inactive_and_handles_constant():
    x = np.array([5.0, 1.0, 3.0, 100.0])
    active = np.array([True, True, True, False])
    out = _minmax_active(x, active)
    assert out[:3] == pytest.approx([1.0, 0.0, 0.5])
    assert np.all(_minmax_active(np.ones(3), np.ones(3, bool)) == 0)


def test_normalized_term_of_a_view_is_invariant_to_its_weight_scale():
    """With one view, score(w=1, lam=1/2, beta=1) = 0.5 Rel - 0.5 KL~ + UCB~ and
    score(w=2, lam=1/3, beta=2/3) = (2/3) Rel - (2/3) KL~ + (4/3) UCB~, which is the first
    multiplied by 4/3; the normalized terms KL~ and UCB~ are the same in both, so the lists match."""
    cands = _candidates(15)
    prof = {"topic": _profile(6)}
    a = eac_rerank(cands, prof, MAPS, VOCABS, {"topic": 1.0}, 0.5, 1.0, 10, normalize=True)
    l = 1 / 3
    b = (1 - l)
    c = eac_rerank(cands, prof, MAPS, VOCABS, {"topic": 2.0}, l, b, 10, normalize=True)
    assert [x for x, _ in a] == [x for x, _ in c]
    assert len({x for x, _ in a}) == 10


def test_beta_zero_ignores_ucb_mode():
    cands = _candidates(15)
    prof = {v: _profile(len(VOCABS[v])) for v in VOCABS}
    w = {v: 1 / 3 for v in VOCABS}
    ref = eac_rerank(cands, prof, MAPS, VOCABS, w, 0.5, 0.0, 10)
    for mode in ("hist", "list"):
        assert eac_rerank(cands, prof, MAPS, VOCABS, w, 0.5, 0.0, 10, ucb_mode=mode) == ref


def test_first_position_has_no_exploration_bonus():
    cands = _candidates(15)
    prof = {v: _profile(len(VOCABS[v])) for v in VOCABS}
    w = {v: 1 / 3 for v in VOCABS}
    first_trad = eac_rerank(cands, prof, MAPS, VOCABS, w, 0.5, 0.0, 1)[0][0]
    for beta in (1.0, 100.0, math.inf):
        for mode in ("full", "hist", "list"):
            assert eac_rerank(cands, prof, MAPS, VOCABS, w, 0.5, beta, 1, ucb_mode=mode)[0][0] == first_trad


def test_beta_infinity_is_the_limit_of_large_beta():
    cands = _candidates(20)
    prof = {"entity": _profile(len(VOCABS["entity"]))}
    w = {"entity": 1.0}
    big = eac_rerank(cands, prof, MAPS, VOCABS, w, 0.5, 1e9, 10, normalize=False)
    inf = eac_rerank(cands, prof, MAPS, VOCABS, w, 0.5, math.inf, 10, normalize=False)
    assert [a for a, _ in big] == [a for a, _ in inf]


def test_unknown_ucb_mode_raises():
    with pytest.raises(ValueError):
        eac_rerank(_candidates(5), {"topic": _profile(6)}, MAPS, VOCABS, {"topic": 1.0}, 0.5, 1.0, 3, ucb_mode="x")


# --------------------------------------------------------------------------- baselines
E = {a: rng.normal(size=300) for a in range(N_ART)}


def test_mmr_lambda_zero_is_relevance_order():
    cands = _candidates(12)
    assert [a for a, _ in mmr(cands, 0.0, 10, E)] == [a for a, _ in cands[:10]]


def test_xquad_lambda_zero_is_relevance_order_and_covers_topics_at_one():
    cands = _candidates(12)
    tidx = {t: i for i, t in enumerate(VOCABS["topic"])}
    p = np.ones(6) / 6
    assert [a for a, _ in xquad(cands, 0.0, 10, MAPS["topic"], tidx, p)] == [a for a, _ in cands[:10]]
    out = [a for a, _ in xquad(cands, 1.0, 3, MAPS["topic"], tidx, p)]
    covered = [set(MAPS["topic"][a]) for a in out]
    assert covered[1] - covered[0]  # the second pick adds a new topic


def test_dpp_returns_k_distinct_items():
    cands = _candidates(15)
    out = [a for a, _ in dpp(cands, 0.5, 10, E)]
    assert len(out) == 10 and len(set(out)) == 10


def test_dedup_tau_above_one_is_relevance_order():
    cands = _candidates(12)
    assert [a for a, _ in dedup(cands, 1.01, 10, MAPS["topic"], MAPS["entity"])] == [a for a, _ in cands[:10]]


# --------------------------------------------------------------------------- position metrics
def test_position_weights_are_the_ndcg_discount():
    assert position_weights(3) == pytest.approx([1.0, 1 / math.log2(3), 0.5])


def test_weighted_variants_reduce_to_set_metrics_without_weights():
    ids = list(range(10))
    w1 = [1.0] * 10
    assert weighted_err(ids, MAPS["entity"], w1) == pytest.approx(weighted_err(ids, MAPS["entity"]))
    assert weighted_tci(ids, MAPS["topic"], 6, w1) == pytest.approx(weighted_tci(ids, MAPS["topic"], 6))


def test_moving_a_repeated_topic_to_the_end_lowers_weighted_tci_only():
    tmap = {0: ["x"], 1: ["x"], 2: ["y"], 3: ["z"]}
    front, back = [0, 1, 2, 3], [0, 2, 3, 1]
    w = position_weights(4)
    assert weighted_tci(front, tmap, 10) == pytest.approx(weighted_tci(back, tmap, 10))
    assert weighted_tci(back, tmap, 10, w) < weighted_tci(front, tmap, 10, w)


def test_category_entropy_and_ild_on_known_list():
    cmap = {0: ["a"], 1: ["b"], 2: ["a"], 3: ["b"]}
    ild, ent = category_ild_entropy([0, 1, 2, 3], cmap)
    assert ild == pytest.approx(4 / 6)
    assert ent == pytest.approx(math.log(2))


def test_calibration_error_is_minimal_for_a_list_matching_the_profile():
    vocab = ["a", "b"]
    amap = {0: ["a"], 1: ["b"], 2: ["a"], 3: ["a"]}
    p = np.array([0.5, 0.5])
    assert calibration_error([0, 1], p, amap, vocab) == pytest.approx(0.0, abs=1e-12)
    assert calibration_error([0, 2, 3], p, amap, vocab) > calibration_error([0, 1], p, amap, vocab)


def test_float_noise_is_treated_as_ties():
    d = np.array([1e-16, -1e-16, 0.0, 2e-17])
    assert wilcoxon_p(d) == 1.0
    assert cohens_d(d) == 0.0
