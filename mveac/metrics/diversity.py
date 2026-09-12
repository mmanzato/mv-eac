"""Category-level diversity metrics: Intra-List Diversity (ILD@K) and Shannon Entropy@K.

Both are computed over the top-K reranked list and describe how varied the
list is *within itself* -- as opposed to the concentration metrics in
``mveac.metrics.concentration``, which describe how the list's own semantic
distribution concentrates onto a small number of values.
"""
from __future__ import annotations

import numpy as np


def intra_list_diversity(
    ranked_ids: list[int],
    category_vectors: dict[int, np.ndarray],
    k: int,
) -> float:
    """Mean pairwise cosine *distance* (1 - cosine similarity) between the
    (multi-)hot category vectors of the top-K items. 0 = every pair identical,
    1 = every pair orthogonal."""
    items = ranked_ids[:k]
    if len(items) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(items)):
        vi = category_vectors.get(items[i])
        if vi is None:
            continue
        for j in range(i + 1, len(items)):
            vj = category_vectors.get(items[j])
            if vj is None:
                continue
            denom = float(np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_sim = float(np.dot(vi, vj) / denom) if denom > 0 else 0.0
            total += 1.0 - cos_sim
            count += 1
    return total / count if count > 0 else 0.0


def list_entropy(
    ranked_ids: list[int],
    article_map: dict[int, list[str]],
    vocab: list[str],
    k: int,
) -> float:
    """Shannon entropy (nats) of the top-K list's distribution over ``vocab``.

    Higher = more evenly spread across values; 0 = every item shares a single
    value. Used for category (paper's Entropy@K) as well as for computing
    view-level entropy for topic/entity/sentiment when reporting diagnostics.
    """
    top_k = ranked_ids[:k]
    idx = {v: i for i, v in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)
    for aid in top_k:
        for value in article_map.get(aid, ()):
            j = idx.get(value)
            if j is not None:
                counts[j] += 1.0
    total = counts.sum()
    if total == 0:
        return 0.0
    q = counts[counts > 0] / total
    return float(-np.sum(q * np.log(q)))


def jaccard_ild(
    ranked_ids: list[int],
    article_map: dict[int, list[str]],
    k: int,
) -> float:
    """Mean pairwise Jaccard *distance* between top-K items' label sets under one view.

    Used for multi-label views (entity, topic) where a cosine-based ILD is
    less natural than set-overlap distance.
    """
    items = ranked_ids[:k]
    if len(items) < 2:
        return 0.0
    label_sets = [set(article_map.get(aid, ())) for aid in items]
    total, count = 0.0, 0
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            a, b = label_sets[i], label_sets[j]
            if not a and not b:
                similarity = 1.0
            else:
                union = len(a | b)
                similarity = len(a & b) / union if union > 0 else 0.0
            total += 1.0 - similarity
            count += 1
    return total / count if count > 0 else 0.0
