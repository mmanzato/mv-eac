"""
KL-divergence calibration helpers (Steck, 2018). NOTE: the rerankers themselves
(``mveac.calibration.eac`` and ``mveac.calibration.mcf``) use the per-element
form p*log((p+eps)/(q+eps)) of the paper's Eq. 1; ``kl_divergence`` below
(renormalized smoothing of q) is a reference implementation for analysis and
tests, numerically equivalent for p > 0 up to O(eps).
"""
from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float) -> float:
    """KL(P || Q) = sum_x p(x) log(p(x) / q(x)), with additive smoothing on Q.

    Only additively smooths and renormalizes ``Q``: ``P`` is a genuine
    probability distribution over the user's own history and is never zero
    where it matters (terms with :math:`p(x)=0` contribute 0 to the sum and
    are skipped rather than smoothed).
    """
    q_smooth = q + epsilon
    q_smooth = q_smooth / q_smooth.sum()
    mask = p > 0
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q_smooth[mask])))


def minmax_normalize(scores: list[float]) -> list[float]:
    """Scale a list of raw relevance scores to [0, 1]; ties-to-1 if the list is constant."""
    arr = np.asarray(scores, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr - lo) / (hi - lo)).tolist()
    return [1.0] * len(scores)
