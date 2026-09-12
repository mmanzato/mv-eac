"""
MCF -- Minimum-Cost-Flow calibration baseline.

Reference: Abdollahpouri, Nazari, Gain, Gibson, Dimakopoulou, Anderton,
Carterette, Lalmas & Jebara (2023), "Calibrated Recommendations as a
Minimum-Cost Flow Problem", WSDM'23. https://doi.org/10.1145/3539597.3570402

MCF solves *exactly* (via network flow) a calibration objective of the same
family as Traditional calibration / EAC's deterministic term, rather than
constructing the list greedily. It is used in the paper strictly as an
empirical control: comparing it to Trad-Cal isolates how much of EAC's
advantage over Trad-Cal is due to *exact* optimization of the calibration
objective (which MCF alone would also capture) versus the *exploration* term
EAC adds on top (which MCF has no equivalent of).

Three design choices, made so that MCF solves a genuinely comparable
objective rather than a different one dressed up as a baseline (see paper
Section 4.4 / RQ1 for how these are reflected in the reported conclusions):

1.  **Calibration direction.** The original paper penalizes
    ``KL(list || user)``. This project's own Trad-Cal/EAC (matching Steck,
    2018) instead uses ``KL(user || list)``, with the same epsilon-smoothing
    convention throughout (``mveac.calibration.traditional.kl_divergence``).
    We implement MCF against *our* direction and epsilon convention so the
    comparison is to an exact solver for the *same* objective, not a
    different one.

2.  **Presentation order is a heuristic, not part of the optimized objective.**
    Trad-Cal/EAC's greedy score has no term that depends on an item's
    position within the list, only on whether it is selected. To keep MCF an
    exact solver for that same "set selection" objective (rather than
    injecting a slot-specific value matrix the greedy score never uses), MCF
    selects the optimal *set* of K items via a single-commodity min-cost
    flow, then orders the selected set for presentation (which NDCG is
    sensitive to) by each item's own
    ``affinity(i) = (1-lambda)*rel_norm(i) + lambda*max_k p_u(k)`` over the
    item's own labels -- the same (1-lambda)/lambda trade-off the objective
    itself optimizes, applied to a single item in isolation (the list-level
    KL term has no meaning for one item alone). This is a display-only
    heuristic: it never changes *which* K items are selected, only how the
    fixed selected set is shown.

3.  **Multi-label views (topic, entity): a single-assignment relaxation.**
    A unit of flow through an item's graph node can only take one outgoing
    edge, so a multi-label item can register against only *one* of its
    labels' cost ladders when selected, never all of them at once (any item
    with several simultaneous labels would need to "duplicate" its flow unit,
    which violates flow conservation). MCF is therefore an *exact* solver for
    the single-label views (category, sentiment) but only an *approximate*
    one for the multi-label views (topic, entity); this is stated explicitly
    wherever MCF-Topic/MCF-Entity results are reported. The final selected
    list is still scored with the true multi-label ERR@K/TCI@K/ILD@K metrics
    like every other method -- the relaxation affects only what the solver
    optimizes internally while building the list.

Requires the ``ortools`` package (``pip install ortools``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ortools.graph.python import min_cost_flow

from mveac.calibration.traditional import minmax_normalize

KL_EPSILON = 1e-10
COST_SCALE = 10**6  # ortools costs must be integers; scale floats before rounding


def _kl_term(p_k: float, q_k: float) -> float:
    """One element of the KL sum: p_k * log((p_k+eps)/(q_k+eps))."""
    return p_k * np.log((p_k + KL_EPSILON) / (q_k + KL_EPSILON))


def _marginal_costs(p_k: float, n: int) -> list[float]:
    """Marginal cost of the i-th unit of flow assigned to value ``k``, for i=1..n.

    ``term(i) = p_k * log((p_k+eps)/(i/n+eps))`` is convex and non-increasing
    in ``i``, so its marginal costs (this function's return value) are
    non-decreasing in ``i`` -- exactly the convexity min-cost flow requires
    to reproduce this penalty exactly via a ladder of unit-capacity edges.
    """
    cumulative = [_kl_term(p_k, i / n) for i in range(0, n + 1)]
    return [cumulative[i] - cumulative[i - 1] for i in range(1, n + 1)]


def mcf_rerank(
    candidates: list[tuple[int, float]],
    user_profile: np.ndarray,
    article_map: dict[int, list[str]],
    vocab: list[str],
    lam: float,
    k: int,
) -> list[tuple[int, float]]:
    """Exact min-cost-flow solution for one semantic view's calibration objective.

        maximize   (1-lam) * sum_{i in S} rel_norm(i)
                    - lam   * sum_x p(x) * log((p(x)+eps) / (count_x(S)/n + eps))
        subject to |S| = n = min(k, n_candidates)

    Returns a list of ``(article_id, synthetic_score)`` of length ``n``, ordered
    by descending per-item affinity among the selected set (see module
    docstring, point 2).
    """
    candidate_ids = [cid for cid, _ in candidates]
    n_candidates = len(candidate_ids)
    n = min(k, n_candidates)
    if n_candidates == 0:
        return []

    relevance = minmax_normalize([s for _, s in candidates])
    vocab_index = {v: i for i, v in enumerate(vocab)}
    item_labels = [[v for v in article_map.get(cid, ()) if v in vocab_index] for cid in candidate_ids]
    active_vocab = sorted({v for labels in item_labels for v in labels})

    # --- Graph: source -> hub -> item nodes -> per-value "count ladder" nodes -> sink ---
    SOURCE, HUB = 0, 1
    item_node = {cid: 2 + i for i, cid in enumerate(candidate_ids)}
    next_id = 2 + n_candidates
    count_node: dict[tuple[str, int], int] = {}
    for value in active_vocab:
        for rank in range(1, n + 1):
            count_node[(value, rank)] = next_id
            next_id += 1
    SINK = next_id

    solver = min_cost_flow.SimpleMinCostFlow()
    solver.add_arc_with_capacity_and_unit_cost(SOURCE, HUB, n, 0)

    for cid, rel, labels in zip(candidate_ids, relevance, item_labels):
        item_cost = -int(round((1.0 - lam) * rel * COST_SCALE))
        solver.add_arc_with_capacity_and_unit_cost(HUB, item_node[cid], 1, item_cost)
        if not labels:
            # No membership under this view: free to select without affecting any
            # value's count.
            solver.add_arc_with_capacity_and_unit_cost(item_node[cid], SINK, 1, 0)
            continue
        for value in labels:
            for rank in range(1, n + 1):
                solver.add_arc_with_capacity_and_unit_cost(item_node[cid], count_node[(value, rank)], 1, 0)

    profile = np.asarray(user_profile, dtype=np.float64)
    for value in active_vocab:
        p_k = float(profile[vocab_index[value]])
        marginal = _marginal_costs(p_k, n)
        for rank in range(1, n + 1):
            cost = int(round(lam * marginal[rank - 1] * COST_SCALE))
            solver.add_arc_with_capacity_and_unit_cost(count_node[(value, rank)], SINK, 1, cost)

    solver.set_node_supply(SOURCE, n)
    solver.set_node_supply(SINK, -n)

    relevance_of = dict(zip(candidate_ids, relevance))
    affinity_of = {
        cid: (1.0 - lam) * relevance_of[cid]
        + lam * (max(float(profile[vocab_index[v]]) for v in labels) if labels else 0.0)
        for cid, labels in zip(candidate_ids, item_labels)
    }

    status = solver.solve()
    if status != solver.OPTIMAL:
        # The construction above is always feasible, so this should not trigger;
        # fall back to a plain affinity ranking just in case.
        ordered = sorted(candidate_ids, key=lambda cid: -affinity_of[cid])[:n]
        return [(cid, float(n - i)) for i, cid in enumerate(ordered)]

    selected: set[int] = set()
    for arc in range(solver.num_arcs()):
        if solver.flow(arc) > 0 and solver.tail(arc) == HUB:
            head = solver.head(arc)
            for cid, node in item_node.items():
                if node == head:
                    selected.add(cid)
                    break

    # Order by descending affinity for presentation (position was not part of the
    # optimized objective); ties break by the base recommender's own candidate
    # order, not Python set-iteration order.
    in_original_order = [cid for cid in candidate_ids if cid in selected]
    ordered = sorted(in_original_order, key=lambda cid: -affinity_of[cid])
    return [(cid, float(len(ordered) - i)) for i, cid in enumerate(ordered)]


def rerank_all_mcf(
    base_scores: dict[int, list[tuple[int, float]]],
    behaviors: pd.DataFrame,
    user_profiles: dict[str, dict[int, np.ndarray]],
    article_maps: dict[str, dict[int, list[str]]],
    vocabs: dict[str, list[str]],
    view: str,
    lam: float,
    k: int,
) -> dict[int, list[tuple[int, float]]]:
    """Apply MCF to every impression for a single semantic view."""
    impression_to_user = dict(zip(behaviors["impression_id"].astype(int), behaviors["user_id"].astype(int)))
    vocab = vocabs[view]
    article_map = article_maps[view]
    profiles = user_profiles[view]

    results: dict[int, list[tuple[int, float]]] = {}
    for imp_id, candidates in base_scores.items():
        imp_id = int(imp_id)
        uid = impression_to_user.get(imp_id)
        profile = profiles.get(uid) if uid is not None else None
        if profile is None:
            profile = np.ones(len(vocab)) / len(vocab)
        results[imp_id] = mcf_rerank(candidates, profile, article_map, vocab, lam, k)
    return results
