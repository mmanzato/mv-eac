"""
Aggregation glue: combine the individual metric functions in ``mveac.metrics``
into whole-evaluation summaries, either averaged over all impressions
(``compute_all``, used for the validation grid search) or kept one row per
impression (``per_impression``, used for the final test-set evaluation and
every statistical test in the paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mveac.metrics.accuracy import ndcg_at_k
from mveac.metrics.concentration import (
    entity_repetition_rate,
    herfindahl_index,
    topic_concentration_index,
)
from mveac.metrics.diversity import intra_list_diversity, jaccard_ild, list_entropy


def _impression_lookup(behaviors: pd.DataFrame) -> pd.DataFrame:
    return behaviors.set_index("impression_id")


def per_impression(
    ranked: dict[int, list[tuple[int, float]]],
    behaviors: pd.DataFrame,
    article_maps: dict[str, dict[int, list[str]]],
    vocabs: dict[str, list[str]],
    category_vectors: dict[int, np.ndarray],
    k: int,
) -> dict[int, dict[str, float]]:
    """Compute every reported metric for every impression in ``ranked``.

    Returns ``{impression_id: {"NDCG@K": ..., "ILD@K": ..., "Entropy": ...,
    "ERR@K": ..., "TCI@K": ..., "HHI@K": ..., "n_topics_present": ...}}``.
    This is the function the whole paper's Wilcoxon/Cohen's-d testing runs on
    (``mveac.evaluation.stats``), since every test needs one paired value per
    impression, not a single averaged number.
    """
    beh = _impression_lookup(behaviors)
    topic_map, entity_map, category_map = article_maps["topic"], article_maps["entity"], article_maps["category"]
    n_topic_vocab = len(vocabs["topic"])

    out: dict[int, dict[str, float]] = {}
    for imp_id, items in ranked.items():
        imp_id = int(imp_id)
        if imp_id not in beh.index:
            continue
        row = beh.loc[imp_id]
        clicked = set(int(x) for x in row["article_ids_clicked"])
        ranked_ids = [aid for aid, _ in items]
        top_k = ranked_ids[:k]

        out[imp_id] = {
            "NDCG@K": ndcg_at_k(ranked_ids, clicked, k),
            "ILD@K": intra_list_diversity(top_k, category_vectors, k),
            "Entropy": list_entropy(top_k, category_map, vocabs["category"], k),
            "ERR@K": entity_repetition_rate(top_k, entity_map, k),
            "TCI@K": topic_concentration_index(top_k, topic_map, n_topic_vocab, k),
            "HHI@K": herfindahl_index(top_k, topic_map, k),
            "ILD_entity@K": jaccard_ild(top_k, entity_map, k),
            "ILD_topic@K": jaccard_ild(top_k, topic_map, k),
            "n_topics_present": len(set(t for aid in top_k for t in topic_map.get(aid, ()))),
        }
    return out


def compute_all(
    ranked: dict[int, list[tuple[int, float]]],
    behaviors: pd.DataFrame,
    article_maps: dict[str, dict[int, list[str]]],
    vocabs: dict[str, list[str]],
    category_vectors: dict[int, np.ndarray],
    k: int,
) -> dict[str, float]:
    """Mean of every metric across all impressions in ``ranked`` -- one row of the
    validation grid search."""
    per_imp = per_impression(ranked, behaviors, article_maps, vocabs, category_vectors, k)
    if not per_imp:
        return {}
    keys = next(iter(per_imp.values())).keys()
    return {key: float(np.mean([row[key] for row in per_imp.values()])) for key in keys}
