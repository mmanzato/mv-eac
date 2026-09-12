"""Ranking-accuracy metric: NDCG@K with binary (clicked/not-clicked) relevance."""
from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked_ids: list[int], clicked_ids: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at cutoff K, binary relevance.

    ``DCG@K = sum_{i=1}^{K} rel_i / log2(i+1)``, normalized by the ideal DCG
    (all clicked articles ranked first). Returns 0.0 if the impression has no
    clicked articles among its candidates.
    """
    top_k = ranked_ids[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, aid in enumerate(top_k) if aid in clicked_ids)
    n_relevant = min(len(clicked_ids), k)
    if n_relevant == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return float(dcg / idcg) if idcg > 0 else 0.0
