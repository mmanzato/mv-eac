"""ItemKNN-W2V: cosine-similarity recommender over pre-trained Word2Vec article embeddings.

No matrix factorization and no training beyond L2-normalizing the fixed
article embeddings: a user is represented by the mean embedding of their
clicked history, and candidates are scored by cosine similarity to that
vector. This makes it a cheap, embedding-only baseline distinct from both
Most Popular (no personalization at all) and NRMS (a trained neural model).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class ItemKNN:
    name = "itemknn"

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None   # (n_articles, dim), L2-normalized
        self._id2idx: dict[int, int] = {}

    def fit(self, embedding_matrix: np.ndarray, id2idx: dict[int, int]) -> "ItemKNN":
        """``embedding_matrix``: (n_articles, dim); ``id2idx``: article_id -> row index."""
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = embedding_matrix / norms
        self._id2idx = id2idx
        log.info("ItemKNN fitted: %d articles, dim=%d", len(id2idx), embedding_matrix.shape[1])
        return self

    def _user_vector(self, history: list[int]) -> np.ndarray | None:
        vecs = [self._embeddings[self._id2idx[aid]] for aid in history if aid in self._id2idx]
        return _l2_normalize(np.mean(vecs, axis=0)) if vecs else None

    def score(self, history: list[int], article_ids: list[int]) -> list[tuple[int, float]]:
        """Cosine similarity between the user vector and each candidate's embedding.

        Falls back to an all-zero score (a neutral tie, broken by the base
        recommender's own candidate order) when the user has no usable history.
        """
        user_vec = self._user_vector(history)
        if user_vec is None:
            return [(aid, 0.0) for aid in article_ids]
        scored = [
            (aid, float(np.dot(user_vec, self._embeddings[self._id2idx[aid]])) if aid in self._id2idx else 0.0)
            for aid in article_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def score_all_impressions(
        self, behaviors: pd.DataFrame, user_history: dict[int, list[int]]
    ) -> dict[int, list[tuple[int, float]]]:
        results: dict[int, list[tuple[int, float]]] = {}
        for row in behaviors.itertuples(index=False):
            hist = user_history.get(int(row.user_id), [])
            results[int(row.impression_id)] = self.score(hist, list(row.article_ids_inview))
        log.info("ItemKNN scored %d impressions", len(results))
        return results
