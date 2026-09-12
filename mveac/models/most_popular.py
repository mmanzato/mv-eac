"""Most Popular baseline: score = global click count in the training behaviors."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class MostPopular:
    """Non-personalized baseline. Every user sees the same ranking of candidates."""

    name = "most_popular"

    def __init__(self) -> None:
        self._click_counts: dict[int, int] = {}

    def fit(self, click_counts: dict[int, int]) -> "MostPopular":
        """``click_counts``: ``article_id -> total clicks`` in the training behaviors."""
        self._click_counts = click_counts
        return self

    def score(self, article_ids: list[int]) -> list[tuple[int, float]]:
        """Score and sort a candidate list by descending popularity."""
        scored = [(aid, float(self._click_counts.get(aid, 0))) for aid in article_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def score_all_impressions(self, behaviors: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
        """Score every impression's candidate set (``article_ids_inview``)."""
        results: dict[int, list[tuple[int, float]]] = {}
        for row in behaviors.itertuples(index=False):
            results[int(row.impression_id)] = self.score(list(row.article_ids_inview))
        log.info("MostPopular scored %d impressions", len(results))
        return results
