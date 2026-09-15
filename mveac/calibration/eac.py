"""
Exploration-Aware Calibration (EAC) and Multi-View EAC (MV-EAC): the paper's
core contribution.

Both are greedy, position-by-position rerankers. At each of the K steps, every
remaining candidate article ``i`` is scored as

    Score(i) = (1 - lambda) * Rel(i)
               - lambda      * sum_v w_v * KL(P_u^v || Q_{L+i}^v)
               + beta        * sum_v w_v * UCB_v(i)

where ``L`` is the partial list built so far, ``Q_{L+i}^v`` is the view-``v``
distribution of ``L`` with candidate ``i`` tentatively appended, ``Rel(i)`` is
the base recommender's own (min-max normalized) relevance score, and
``UCB_v(i)`` is a per-view UCB-style exploration bonus (Eq. 6-7 of the paper):
elements the user's own profile under-weights but that have been shown least
often so far in the constructed list receive a larger bonus, encouraging the
reranker to actively probe underexplored regions of the user's own profile
rather than only exploiting it.

For single-view methods (``method in {"category_eac", "entity_eac",
"topic_eac", "sentiment_eac"}``) there is exactly one view ``v`` and the sums
above collapse to their single term; ``"traditional"`` is the special case
``beta=0`` (no exploration term at all -- deterministic KL calibration only,
Steck 2018). For ``"mv_eac"``, several views are combined via user-supplied
weights ``w_v`` (the paper's :math:`\\mathcal{V}^\\star` and its weight vector,
Section 3.5).

Implementation note: both rerankers are fully vectorized with numpy -- at each
of the K steps, every remaining candidate is scored via broadcasting rather
than a Python loop, since the article x semantic-vocabulary matrix (up to
5,852 entity dimensions) makes a naive per-candidate Python loop the dominant
cost of the whole pipeline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mveac.calibration.traditional import minmax_normalize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared building block: article x vocabulary indicator matrix
# ---------------------------------------------------------------------------

def _build_article_matrix(
    candidate_ids: list[int],
    article_map: dict[int, list[str]],
    value_index: dict[str, int],
    vocab_size: int,
) -> np.ndarray:
    """(n_candidates, vocab_size) count matrix: how many times each candidate carries
    each vocabulary value (usually 0 or 1, but topic/entity articles can repeat a value)."""
    matrix = np.zeros((len(candidate_ids), vocab_size), dtype=np.float32)
    for row, aid in enumerate(candidate_ids):
        for value in article_map.get(aid, ()):
            col = value_index.get(value)
            if col is not None:
                matrix[row, col] += 1.0
    return matrix


def _kl_all(profile: np.ndarray, candidate_distributions: np.ndarray, epsilon: float) -> np.ndarray:
    """Vectorized KL(profile || candidate_distributions[i]) for every row i."""
    return np.sum(
        profile * np.log((profile + epsilon) / (candidate_distributions + epsilon)), axis=1
    )


# ---------------------------------------------------------------------------
# Single-view greedy reranker (Traditional calibration when beta=0, EAC otherwise)
# ---------------------------------------------------------------------------

def _greedy_rerank_single_view(
    candidates: list[tuple[int, float]],
    user_profile: np.ndarray,
    article_map: dict[int, list[str]],
    vocab: list[str],
    lam: float,
    beta: float,
    k: int,
    epsilon: float,
) -> list[tuple[int, float]]:
    vocab_size = len(vocab)
    value_index = {v: i for i, v in enumerate(vocab)}
    candidate_ids = [cid for cid, _ in candidates]
    n_candidates = len(candidate_ids)

    article_matrix = _build_article_matrix(candidate_ids, article_map, value_index, vocab_size).astype(np.float64)
    relevance = np.array(minmax_normalize([s for _, s in candidates]), dtype=np.float64)
    n_values_per_article = article_matrix.sum(axis=1)  # UCB averaging denominator per candidate

    list_counts = np.zeros(vocab_size, dtype=np.float64)     # running Q_L numerator
    exposure_counts = np.zeros(vocab_size, dtype=np.float64)  # times each value has appeared so far
    active = np.ones(n_candidates, dtype=bool)
    selected: list[tuple[int, float]] = []
    steps = min(k, n_candidates)
    t = 0  # number of items placed so far (the UCB "round" counter)

    for _ in range(steps):
        candidate_totals = list_counts[None, :] + article_matrix       # Q_{L+i} numerator, all i at once
        denom = candidate_totals.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        candidate_distributions = candidate_totals / denom
        kl_term = _kl_all(user_profile, candidate_distributions, epsilon)

        ucb_term = np.zeros(n_candidates, dtype=np.float64)
        if beta > 0 and t > 0:
            # Eq. 6-7: elements the user profile favors little (1 - p) and that have
            # appeared rarely so far (small exposure_counts) get the largest bonus;
            # a candidate's UCB score is the mean bonus over its own vocabulary values.
            per_value_ucb = (1.0 - user_profile) * np.sqrt(np.log(t + 1) / (exposure_counts + 1))
            numerator = article_matrix @ per_value_ucb
            safe_denom = np.where(n_values_per_article > 0, n_values_per_article, 1.0)
            ucb_term = np.where(n_values_per_article > 0, numerator / safe_denom, 0.0)

        scores = (1.0 - lam) * relevance - lam * kl_term + beta * ucb_term
        scores[~active] = -np.inf

        # np.argmax returns the first maximum: ties go to the candidate ranked higher by the
        # base recommender (candidates arrive in base-rank order; logged order for cold-start users).
        # A candidate with no label under a view contributes no exploration bonus for that view
        # and leaves the list distribution Q of that view unchanged (paper Section 3.3).
        best = int(np.argmax(scores))
        if not active[best]:
            break  # every remaining candidate was already excluded (shouldn't happen)

        selected.append((candidate_ids[best], float(scores[best])))
        list_counts += article_matrix[best]
        exposure_counts += article_matrix[best]
        active[best] = False
        t += 1

    return selected


def traditional_rerank(
    candidates: list[tuple[int, float]],
    user_profile: np.ndarray,
    article_map: dict[int, list[str]],
    vocab: list[str],
    lam: float,
    k: int,
    epsilon: float = 1e-10,
) -> list[tuple[int, float]]:
    """Deterministic KL calibration (Steck, 2018): the beta=0 special case of EAC."""
    return _greedy_rerank_single_view(candidates, user_profile, article_map, vocab, lam, 0.0, k, epsilon)


def single_view_eac(
    candidates: list[tuple[int, float]],
    user_profile: np.ndarray,
    article_map: dict[int, list[str]],
    vocab: list[str],
    lam: float,
    beta: float,
    k: int,
    epsilon: float = 1e-10,
) -> list[tuple[int, float]]:
    """Exploration-Aware Calibration over a single semantic view."""
    return _greedy_rerank_single_view(candidates, user_profile, article_map, vocab, lam, beta, k, epsilon)


# ---------------------------------------------------------------------------
# Multi-view greedy reranker (MV-EAC)
# ---------------------------------------------------------------------------

def multi_view_eac(
    candidates: list[tuple[int, float]],
    user_profiles: dict[str, np.ndarray],
    article_maps: dict[str, dict[int, list[str]]],
    vocabs: dict[str, list[str]],
    weights: dict[str, float],
    lam: float,
    beta: float,
    k: int,
    epsilon: float = 1e-10,
) -> list[tuple[int, float]]:
    """Multi-View EAC: jointly calibrates + explores across several weighted views.

    ``weights`` need not include every key of ``user_profiles``; only views
    with a strictly positive weight contribute to the score (this is how the
    RQ2 ablation and RQ4 weight sweep are implemented -- by zeroing out a
    view's weight rather than removing it from the data structures).
    """
    views = [v for v in user_profiles if weights.get(v, 0.0) > 0]
    candidate_ids = [cid for cid, _ in candidates]
    n_candidates = len(candidate_ids)

    value_index = {v: {val: i for i, val in enumerate(vocabs[v])} for v in views}
    article_matrix = {
        v: _build_article_matrix(candidate_ids, article_maps[v], value_index[v], len(vocabs[v])).astype(np.float64)
        for v in views
    }
    relevance = np.array(minmax_normalize([s for _, s in candidates]), dtype=np.float64)

    list_counts = {v: np.zeros(len(vocabs[v])) for v in views}
    exposure_counts = {v: np.zeros(len(vocabs[v]), dtype=np.float64) for v in views}
    n_values_per_article = {v: article_matrix[v].sum(axis=1) for v in views}

    active = np.ones(n_candidates, dtype=bool)
    selected: list[tuple[int, float]] = []
    steps = min(k, n_candidates)
    t = 0

    for _ in range(steps):
        kl_total = np.zeros(n_candidates, dtype=np.float64)
        ucb_total = np.zeros(n_candidates, dtype=np.float64)

        for v in views:
            w = weights[v]
            candidate_totals = list_counts[v][None, :] + article_matrix[v]
            denom = candidate_totals.sum(axis=1, keepdims=True)
            denom = np.where(denom > 0, denom, 1.0)
            candidate_distributions = candidate_totals / denom
            kl_total += w * _kl_all(user_profiles[v], candidate_distributions, epsilon)

            if beta > 0 and t > 0:
                per_value_ucb = (1.0 - user_profiles[v]) * np.sqrt(np.log(t + 1) / (exposure_counts[v] + 1))
                numerator = article_matrix[v] @ per_value_ucb
                safe_denom = np.where(n_values_per_article[v] > 0, n_values_per_article[v], 1.0)
                ucb_total += w * np.where(n_values_per_article[v] > 0, numerator / safe_denom, 0.0)

        scores = (1.0 - lam) * relevance - lam * kl_total + beta * ucb_total
        scores[~active] = -np.inf

        # np.argmax returns the first maximum: ties go to the candidate ranked higher by the
        # base recommender (candidates arrive in base-rank order; logged order for cold-start users).
        # A candidate with no label under a view contributes no exploration bonus for that view
        # and leaves the list distribution Q of that view unchanged (paper Section 3.3).
        best = int(np.argmax(scores))
        if not active[best]:
            break

        selected.append((candidate_ids[best], float(scores[best])))
        for v in views:
            list_counts[v] += article_matrix[v][best]
            exposure_counts[v] += article_matrix[v][best]
        active[best] = False
        t += 1

    return selected


# ---------------------------------------------------------------------------
# Dispatch: rerank every impression for a given method
# ---------------------------------------------------------------------------

_SINGLE_VIEW_METHODS = {
    "category_eac": "category",
    "subcategory_eac": "subcategory",
    "entity_eac": "entity",
    "topic_eac": "topic",
    "sentiment_eac": "sentiment",
    "traditional": "category",  # the paper's Trad-Cal-Cat baseline
}


def rerank_all(
    base_scores: dict[int, list[tuple[int, float]]],
    behaviors: pd.DataFrame,
    user_profiles: dict[str, dict[int, np.ndarray]],
    article_maps: dict[str, dict[int, list[str]]],
    vocabs: dict[str, list[str]],
    method: str,
    lam: float,
    beta: float,
    k: int,
    weights: dict[str, float] | None = None,
    epsilon: float = 1e-10,
) -> dict[int, list[tuple[int, float]]]:
    """Apply one reranking method to every impression in ``behaviors``.

    ``method`` is one of ``"original"`` (pass the base recommender's own
    ranking through unchanged), ``"traditional"`` (single-view Trad-Cal on
    category), one of the single-view EAC methods in ``_SINGLE_VIEW_METHODS``,
    or ``"mv_eac"`` (Multi-View EAC; ``weights`` defaults to
    ``mveac.config.MV_UNIFORM_WEIGHTS`` when omitted).

    A user missing a profile for the relevant view (e.g. no reading history)
    falls back to the uniform distribution over that view's vocabulary --
    KL(uniform || Q) then measures only how far the list's own distribution
    is from uniform, a neutral calibration target for a cold-start user.
    """
    if weights is None:
        from mveac.config import MV_UNIFORM_WEIGHTS
        weights = MV_UNIFORM_WEIGHTS

    impression_to_user = dict(zip(behaviors["impression_id"].astype(int), behaviors["user_id"].astype(int)))
    results: dict[int, list[tuple[int, float]]] = {}

    for imp_id, candidates in base_scores.items():
        imp_id = int(imp_id)
        uid = impression_to_user.get(imp_id)
        if uid is None or method == "original":
            results[imp_id] = candidates[:k]
            continue

        if method == "mv_eac":
            profiles = {
                v: user_profiles[v].get(uid, np.ones(len(vocabs[v])) / len(vocabs[v]))
                for v in user_profiles
            }
            results[imp_id] = multi_view_eac(
                candidates, profiles, article_maps, vocabs, weights, lam, beta, k, epsilon
            )
            continue

        view = _SINGLE_VIEW_METHODS.get(method)
        if view is None:
            raise ValueError(f"Unknown reranking method: {method!r}")

        profile = user_profiles[view].get(uid, np.ones(len(vocabs[view])) / len(vocabs[view]))
        if method == "traditional":
            results[imp_id] = traditional_rerank(candidates, profile, article_maps[view], vocabs[view], lam, k, epsilon)
        else:
            results[imp_id] = single_view_eac(candidates, profile, article_maps[view], vocabs[view], lam, beta, k, epsilon)

    return results
