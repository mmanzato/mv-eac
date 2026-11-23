"""
Position-sensitive metrics and per-view calibration error (revised paper, Section 4.5.4).

ILD@K, Entropy@K, ERR@K and TCI@K depend only on the *set* of articles in a list. The
functions below make them sensitive to what the user sees first:

* top-5 variants: call the set metric on ``ranked_ids[:5]``;
* position-weighted variants: the article at rank r (1-based) has weight
  ``w_r = 1 / log2(r + 1)`` (the NDCG discount). In Entropy_pw and TCI_pw each label
  exposure counts ``w_r``; in ERR_pw each entity mention counts ``w_r`` in numerator and
  denominator; in ILD_pw the pair (i, j) counts ``w_i * w_j``.

``calibration_error`` is the miscalibration C_KL of Steck (2018) between the user's
profile P and the label-exposure distribution Q of the final list, with
``q~ = (1 - alpha) Q + alpha P`` and ``alpha = 0.01``.

All functions take the article maps used everywhere else in the package
(``article_id -> list of labels``); ``category_map`` is single-label.
"""
from __future__ import annotations

import math

import numpy as np


def position_weights(n: int) -> list[float]:
    """NDCG discount 1/log2(r+1) for ranks r = 1..n."""
    return [1.0 / math.log2(r + 2) for r in range(n)]


def category_ild_entropy(
    ranked_ids: list[int], category_map: dict[int, list[str]], weights: list[float] | None = None
) -> tuple[float, float]:
    """(ILD, Shannon entropy) of the category labels, optionally position-weighted.

    ILD is the (weighted) share of article pairs with different categories, the
    single-label cosine distance; articles without a category are skipped."""
    cats = [(category_map.get(a) or [None])[0] for a in ranked_ids]
    w = weights if weights is not None else [1.0] * len(cats)
    num = den = 0.0
    for i in range(len(cats)):
        if cats[i] is None:
            continue
        for j in range(i + 1, len(cats)):
            if cats[j] is None:
                continue
            ww = w[i] * w[j]
            num += ww * (cats[i] != cats[j])
            den += ww
    ild = num / den if den > 0 else 0.0
    counts: dict[str, float] = {}
    for c, x in zip(cats, w):
        if c is not None:
            counts[c] = counts.get(c, 0.0) + x
    total = sum(counts.values())
    entropy = -sum((v / total) * math.log(v / total) for v in counts.values()) if total > 0 else 0.0
    return ild, entropy


def weighted_err(
    ranked_ids: list[int], entity_map: dict[int, list[str]], weights: list[float] | None = None
) -> float:
    """Entity Repetition Rate with each mention weighted by its article's position weight."""
    seen: set[str] = set()
    repeated = total = 0.0
    for r, a in enumerate(ranked_ids):
        x = 1.0 if weights is None else weights[r]
        for e in entity_map.get(a, ()):
            total += x
            if e in seen:
                repeated += x
            seen.add(e)
    return repeated / total if total > 0 else 0.0


def weighted_tci(
    ranked_ids: list[int], topic_map: dict[int, list[str]], vocab_size: int, weights: list[float] | None = None
) -> float:
    """Topic Concentration Index (HHI normalized by the full topic vocabulary), with each
    topic-label exposure weighted by its article's position weight."""
    counts: dict[str, float] = {}
    for r, a in enumerate(ranked_ids):
        x = 1.0 if weights is None else weights[r]
        for t in topic_map.get(a, ()):
            counts[t] = counts.get(t, 0.0) + x
    s = sum(counts.values())
    if s == 0:
        return 0.0
    hhi = sum((v / s) ** 2 for v in counts.values())
    return min(1.0, max(0.0, (hhi - 1 / vocab_size) / (1 - 1 / vocab_size)))


def calibration_error(
    ranked_ids: list[int],
    profile: np.ndarray,
    article_map: dict[int, list[str]],
    vocab: list[str],
    alpha: float = 0.01,
) -> float:
    """Steck's C_KL(P || q~) of one list under one view; lower is better calibrated.

    ``profile`` is the user's distribution over ``vocab`` (uniform for a user without
    history). Q counts each label occurrence of the list's articles, as in the reranker."""
    index = {x: i for i, x in enumerate(vocab)}
    q = np.zeros(len(vocab))
    for a in ranked_ids:
        for x in article_map.get(a, ()):
            j = index.get(x)
            if j is not None:
                q[j] += 1.0
    if q.sum() > 0:
        q /= q.sum()
    m = profile > 0
    q_tilde = (1 - alpha) * q[m] + alpha * profile[m]
    return float(np.sum(profile[m] * np.log(profile[m] / q_tilde)))
