"""
Semantic vocabularies, article maps, and user/global preference profiles.

Five candidate semantic views are extracted from EB-NeRD article metadata:
category, subcategory, entity, topic, and sentiment (``mveac.config.ALL_VIEWS``).
For each view we build:

  * a **vocabulary**: the sorted list of distinct values the view can take;
  * an **article map**: ``article_id -> [values]`` (a list, since entity and
    topic are multi-label -- an article can mention several named entities or
    cover several topics at once, while category, subcategory and sentiment
    are single-label);
  * a **user profile**: for each user, the normalized frequency distribution
    of that view's values across the user's reading history, i.e.
    :math:`P_u^v \\in \\Delta^{|\\mathcal{X}_v|-1}` in the paper's notation
    (Section 3.2). Users with no reading history for a view fall back to the
    uniform distribution.

All profile vectors are plain ``numpy`` arrays over the view's vocabulary
index, in the exact order ``build_all_vocabs`` returns.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article-level maps: article_id -> [values] per view
# ---------------------------------------------------------------------------

def build_article_maps(
    articles: pd.DataFrame,
    entity_vocab: set[str] | None = None,
) -> dict[str, dict[int, list[str]]]:
    """Build ``{view: {article_id: [values]}}`` for all five candidate views.

    ``entity_vocab``, when given, restricts each article's entity list to
    entities that survive the minimum-frequency filter (see
    ``build_entity_vocab``); pass ``None`` to keep every entity.
    """
    maps: dict[str, dict[int, list[str]]] = {
        "category": {}, "subcategory": {}, "entity": {}, "topic": {}, "sentiment": {},
    }
    for row in articles.itertuples(index=False):
        aid = int(row.article_id)
        maps["category"][aid] = [str(row.category_str)]
        maps["subcategory"][aid] = [str(int(s)) for s in (row.subcategory or []) if s is not None]
        entities = list(row.ner_clusters) if row.ner_clusters is not None else []
        maps["entity"][aid] = (
            [e for e in entities if e in entity_vocab] if entity_vocab is not None else entities
        )
        topics = list(row.topics) if row.topics is not None else []
        maps["topic"][aid] = topics
        maps["sentiment"][aid] = [str(row.sentiment_label)]

    log.info(
        "Article maps built: cat=%d subcat=%d entity=%d topic=%d sentiment=%d",
        *(len(maps[d]) for d in ("category", "subcategory", "entity", "topic", "sentiment")),
    )
    return maps


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

def build_entity_vocab(articles: pd.DataFrame, min_freq: int) -> list[str]:
    """Entities appearing in at least ``min_freq`` distinct articles (paper: min_freq=10)."""
    counts: Counter = Counter()
    for entities in articles["ner_clusters"]:
        counts.update(set(entities))
    vocab = sorted(e for e, c in counts.items() if c >= min_freq)
    log.info("Entity vocab: %d entities (min_freq=%d, %d seen at least once)",
              len(vocab), min_freq, len(counts))
    return vocab


def build_topic_vocab(articles: pd.DataFrame) -> list[str]:
    topics: set[str] = set()
    for t in articles["topics"]:
        if t is not None:
            topics.update(str(x) for x in t)
    vocab = sorted(topics)
    log.info("Topic vocab: %d unique topics", len(vocab))
    return vocab


def build_subcategory_vocab(articles: pd.DataFrame) -> list[str]:
    subcats: set[str] = set()
    for sc in articles["subcategory"]:
        if sc is not None:
            subcats.update(str(int(s)) for s in sc if s is not None)
    return sorted(subcats)


def build_category_vocab(articles: pd.DataFrame) -> list[str]:
    return sorted(articles["category_str"].unique().tolist())


def build_all_vocabs(
    articles: pd.DataFrame,
    entity_min_freq: int,
    sentiment_labels: list[str],
) -> dict[str, list[str]]:
    """Build all five view vocabularies at once."""
    return {
        "category": build_category_vocab(articles),
        "subcategory": build_subcategory_vocab(articles),
        "entity": build_entity_vocab(articles, entity_min_freq),
        "topic": build_topic_vocab(articles),
        "sentiment": sentiment_labels,
    }


# ---------------------------------------------------------------------------
# Profile construction (one dimension at a time)
# ---------------------------------------------------------------------------

def _normalized_frequency(
    article_ids: list[int],
    article_map: dict[int, list[str]],
    vocab: list[str],
) -> np.ndarray:
    """Normalized frequency vector over ``vocab`` for a bag of articles.

    Falls back to the uniform distribution when the bag contributes no
    value under this view (e.g. a user with no history, or an article whose
    metadata is missing for this view).
    """
    idx = {v: i for i, v in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)
    for aid in article_ids:
        for val in article_map.get(aid, ()):
            j = idx.get(val)
            if j is not None:
                counts[j] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else np.ones(len(vocab)) / len(vocab)


def build_user_profiles(
    user_history: dict[int, list[int]],
    article_map: dict[int, list[str]],
    vocab: list[str],
) -> dict[int, np.ndarray]:
    """Per-user profile :math:`P_u^v` for a single view."""
    profiles = {uid: _normalized_frequency(hist, article_map, vocab) for uid, hist in user_history.items()}
    log.info("User profiles [vocab=%d]: %d users", len(vocab), len(profiles))
    return profiles


def build_global_profile(
    articles: pd.DataFrame,
    article_map: dict[int, list[str]],
    vocab: list[str],
) -> np.ndarray:
    """Corpus-wide reference distribution :math:`Q_\\text{corpus}^v` for a single view."""
    all_ids = articles["article_id"].astype(int).tolist()
    return _normalized_frequency(all_ids, article_map, vocab)


def build_category_vectors(
    articles: pd.DataFrame,
    article_map: dict[int, list[str]],
    vocab: list[str],
) -> dict[int, np.ndarray]:
    """One-hot (multi-hot) category vector per article, used by ILD@K (cosine distance)."""
    idx = {v: i for i, v in enumerate(vocab)}
    vectors: dict[int, np.ndarray] = {}
    for row in articles.itertuples(index=False):
        aid = int(row.article_id)
        vec = np.zeros(len(vocab), dtype=np.float32)
        for val in article_map.get(aid, ()):
            j = idx.get(val)
            if j is not None:
                vec[j] = 1.0
        vectors[aid] = vec
    return vectors
