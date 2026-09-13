"""
The paper's two proposed metrics: Entity Repetition Rate (ERR@K) and Topic
Concentration Index (TCI@K). Both quantify informational concentration in a
recommendation list at a finer grain than category-level diversity metrics
can see -- a list can be perfectly diverse across editorial categories while
still repeating the same named entities or clustering around one topic.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def entity_repetition_rate(
    ranked_ids: list[int],
    entity_map: dict[int, list[str]],
    k: int,
) -> float:
    """ERR@K = (total entity exposures - unique entities) / total entity exposures.

    Entity annotations are treated as a MULTISET (paper Eq. 13): an entity
    annotated twice in the same article counts as two mentions, so within-article
    repetition also counts. 0 when every entity mention in the top-K list is
    distinct; approaches 1 as the same entities are repeated. Returns 0 when the
    top-K list carries no entity mention at all (<0.005% of lists in EB-NeRD Large).
    """
    top_k = ranked_ids[:k]
    exposures: list[str] = []
    for aid in top_k:
        exposures.extend(entity_map.get(aid, ()))
    total = len(exposures)
    if total == 0:
        return 0.0
    unique = len(set(exposures))
    return (total - unique) / total


def _topic_frequencies(ranked_ids: list[int], topic_map: dict[int, list[str]], k: int) -> Counter:
    """Count topic-label exposures across the top-K list (an article can carry several)."""
    freq: Counter = Counter()
    for aid in ranked_ids[:k]:
        for topic in topic_map.get(aid, ()):
            freq[topic] += 1
    return freq


def topic_concentration_index(
    ranked_ids: list[int],
    topic_map: dict[int, list[str]],
    topic_vocab_size: int,
    k: int,
) -> float:
    """TCI@K: a Herfindahl-Hirschman-Index-based measure of topical concentration.

    Let :math:`\\hat q(x)` be the normalized frequency of topic ``x`` among all
    topic exposures in the top-K list, and
    :math:`\\mathrm{HHI} = \\sum_x \\hat q(x)^2` the (raw) Herfindahl index of
    that distribution. TCI@K normalizes HHI against the *full topic
    vocabulary size* :math:`N = |\\mathcal{X}_\\text{top}|` (78 for EB-NeRD
    Large), not against the number of topics actually present in this
    particular list:

        TCI@K = clip( (HHI - 1/N) / (1 - 1/N),  0, 1 )

    This is a deliberate departure from the more obvious normalization by the
    number of *distinct topics present*, ``n``: that alternative is undefined
    (0/0) for the single-topic list it should flag as maximally concentrated,
    and assigns identical scores to any two lists whose present topics happen
    to be evenly spread, regardless of how many topics that is -- e.g. a list
    spanning 2 topics evenly and a list spanning 30 topics evenly would score
    the same. Normalizing against the fixed vocabulary size N instead keeps
    TCI@K well-defined everywhere and sensitive to how much of the topic space
    a list actually reaches.

    0 = perfectly uniform over the vocabulary; 1 = every topic exposure is the
    same single topic. TCI@K is an affine transform of HHI, so paired effect sizes
    are identical for both. Returns 0 when the top-K list carries no topic labels.
    """
    freq = _topic_frequencies(ranked_ids, topic_map, k)
    total = sum(freq.values())
    if total == 0:
        return 0.0
    shares = np.fromiter(freq.values(), dtype=np.float64, count=len(freq)) / total
    hhi = float((shares**2).sum())
    tci = (hhi - 1.0 / topic_vocab_size) / (1.0 - 1.0 / topic_vocab_size)
    return float(min(1.0, max(0.0, tci)))


def herfindahl_index(ranked_ids: list[int], topic_map: dict[int, list[str]], k: int) -> float:
    """The raw (un-normalized) Herfindahl-Hirschman Index of the top-K topic distribution.

    Reported alongside TCI@K for transparency: unlike TCI@K, HHI does not
    depend on the vocabulary size and is directly comparable to standard
    market-concentration readings from economics
    (Hirschman, 1945; Herfindahl, 1950).
    """
    freq = _topic_frequencies(ranked_ids, topic_map, k)
    total = sum(freq.values())
    if total == 0:
        return 0.0
    shares = np.fromiter(freq.values(), dtype=np.float64, count=len(freq)) / total
    return float((shares**2).sum())
