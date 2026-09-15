"""Instrumented copy of the MV-EAC greedy reranker, for the robustness analyses of the
Supplementary Material (Section S7). Not used by the main pipeline.

``diagnostic_multi_view_eac`` reproduces :func:`mveac.calibration.eac.multi_view_eac`
exactly when ``unlabeled_rule="zero"`` (tested in ``tests/test_greedy_diagnostics.py``) and
adds two options:

* ``unlabeled_rule="max"`` gives a candidate with no label under a view the largest
  exploration bonus Eq. (3) allows, ``sqrt(ln(T+1))``, instead of zero;
* ``record_steps=True`` returns, for every greedy step with at least two remaining
  candidates, the spread (standard deviation across the remaining candidates) of the
  relevance term and of each view's weighted KL and UCB terms, the mean unweighted KL and
  UCB of each view, and whether removing a view's two terms would change the selected article.
"""
from __future__ import annotations

import numpy as np

from mveac.calibration.eac import _build_article_matrix, _kl_all
from mveac.calibration.traditional import minmax_normalize


def diagnostic_multi_view_eac(
    candidates: list[tuple[int, float]],
    user_profiles: dict[str, np.ndarray],
    article_maps: dict[str, dict[int, list[str]]],
    vocabs: dict[str, list[str]],
    weights: dict[str, float],
    lam: float,
    beta: float,
    k: int,
    epsilon: float = 1e-10,
    unlabeled_rule: str = "zero",
    record_steps: bool = False,
):
    if unlabeled_rule not in ("zero", "max"):
        raise ValueError(f"unknown unlabeled_rule: {unlabeled_rule}")
    views = [v for v in user_profiles if weights.get(v, 0.0) > 0]
    candidate_ids = [cid for cid, _ in candidates]
    n = len(candidate_ids)
    index = {v: {val: i for i, val in enumerate(vocabs[v])} for v in views}
    matrix = {v: _build_article_matrix(candidate_ids, article_maps[v], index[v], len(vocabs[v])).astype(np.float64)
              for v in views}
    relevance = np.array(minmax_normalize([s for _, s in candidates]), dtype=np.float64)
    counts = {v: np.zeros(len(vocabs[v])) for v in views}
    n_values = {v: matrix[v].sum(axis=1) for v in views}
    active = np.ones(n, dtype=bool)
    selected, steps = [], []
    t = 0
    for step in range(min(k, n)):
        kl, ucb = {}, {}
        for v in views:
            totals = counts[v][None, :] + matrix[v]
            denom = totals.sum(axis=1, keepdims=True)
            kl[v] = _kl_all(user_profiles[v], totals / np.where(denom > 0, denom, 1.0), epsilon)
            ucb[v] = np.zeros(n)
            if beta > 0 and t > 0:
                per_value = (1.0 - user_profiles[v]) * np.sqrt(np.log(t + 1) / (counts[v] + 1))
                labeled = n_values[v] > 0
                empty_value = np.sqrt(np.log(t + 1)) if unlabeled_rule == "max" else 0.0
                ucb[v] = np.where(labeled, (matrix[v] @ per_value) / np.where(labeled, n_values[v], 1.0), empty_value)
        terms = {"rel": (1.0 - lam) * relevance}
        for v in views:
            terms[f"kl_{v}"] = -lam * weights[v] * kl[v]
            terms[f"ucb_{v}"] = beta * weights[v] * ucb[v]
        scores = sum(terms.values())
        scores[~active] = -np.inf
        best = int(np.argmax(scores))
        if not active[best]:
            break
        remaining = np.where(active)[0]
        if record_steps and len(remaining) > 1:
            row = {"step": step, "n_remaining": int(len(remaining))}
            for name, values in terms.items():
                row[f"sd_{name}"] = float(np.std(values[remaining]))
            for v in views:
                alternative = scores - terms[f"kl_{v}"] - terms[f"ucb_{v}"]
                row[f"flip_{v}"] = int(remaining[np.argmax(alternative[remaining])] != best)
                row[f"mean_kl_{v}"] = float(np.mean(kl[v][remaining]))
                row[f"mean_ucb_{v}"] = float(np.mean(ucb[v][remaining]))
            steps.append(row)
        selected.append((candidate_ids[best], float(scores[best])))
        for v in views:
            counts[v] += matrix[v][best]
        active[best] = False
        t += 1
    return (selected, steps) if record_steps else selected
