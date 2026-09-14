"""
Statistical significance testing: paired Wilcoxon signed-rank tests, Cohen's
d effect sizes, and multiplicity correction across the whole family of
comparisons the paper makes (465 tests in one family across all research
questions, see scripts/09_statistical_tests.py), plus a user-level clustered re-test to check that impression-level
non-independence (a user contributes ~1.7 impressions on average to the test
subsample) is not driving the significance pattern on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(diff: np.ndarray) -> float:
    """Cohen's d_z for a paired difference: mean difference / its own standard deviation.

    NaN/inf values are dropped before computing; returns NaN if fewer than 2
    finite values remain, and 0.0 if the finite values are all identical
    (zero variance).
    """
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def wilcoxon_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for a paired difference.

    Returns 1.0 (no evidence against the null) if there are fewer than 2
    finite, non-identical values -- the test is undefined in that case, and
    treating it as "not significant" is the conservative choice.
    """
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2 or np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down correction. Returns a boolean rejection mask
    aligned with ``p_values`` (True = still significant after correction)."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break  # once one test fails, every later (larger-p) test also fails
    return reject


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate correction. Returns a boolean
    rejection mask aligned with ``p_values``."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = np.zeros(m, dtype=bool)
    largest_significant_rank = 0
    for rank, i in enumerate(order, start=1):
        if p_values[i] <= alpha * rank / m:
            largest_significant_rank = rank
    for rank, i in enumerate(order, start=1):
        if rank <= largest_significant_rank:
            reject[i] = True
    return reject


def compare(
    values_a: pd.Series,
    values_b: pd.Series,
    user_ids: pd.Series,
    minimize: bool = False,
) -> dict[str, float]:
    """One paired comparison (a vs. b) on a single metric, at both the
    impression level and the user-clustered level.

    ``values_a``/``values_b`` must share the same index (impression_id) and
    ``user_ids`` must be aligned with that same index. The user-clustered
    version aggregates to one paired observation per user (the mean of that
    user's impression-level differences) before recomputing the test, which
    both reduces the effective sample size to the number of unique users and
    checks whether a small number of highly-active users could be driving an
    impression-level result on their own.

    Sign convention (the one used in every table of the paper): ``delta``,
    ``d_impression`` and ``d_user`` are computed on the RAW difference ``a - b``.
    For a "higher is better" metric a positive value means ``a`` is better;
    for a "lower is better" metric (ERR@K, TCI@K) a NEGATIVE value means ``a``
    is better. ``d_impression_improvement`` / ``d_user_improvement`` repeat
    the effect sizes with the sign flipped for minimized metrics
    (``minimize=True``), so that positive always means "``a`` is better".
    """
    diff = (values_a.values - values_b.values).astype(float)
    sign = -1.0 if minimize else 1.0

    per_user_diff = pd.Series(diff, index=user_ids.values).groupby(level=0).mean().values
    d_imp, d_usr = cohens_d(diff), cohens_d(per_user_diff)

    return {
        "n_impressions": int(len(diff)),
        "n_users": int(len(per_user_diff)),
        "delta": float(np.mean(diff)),
        "d_impression": d_imp,
        "p_impression": wilcoxon_p(diff),
        "d_user": d_usr,
        "p_user": wilcoxon_p(per_user_diff),
        "d_impression_improvement": sign * d_imp,
        "d_user_improvement": sign * d_usr,
    }


# The paper's own practical-relevance threshold (NOT Cohen's 0.2/0.5/0.8 benchmarks, which were
# proposed for between-group d): |d| below this is treated as negligible.
EFFECT_SIZE_NEGLIGIBLE = 0.10


def is_practically_meaningful(d: float) -> bool:
    """Whether an effect size clears the |d| >= 0.10 bar the paper uses throughout
    to distinguish "true even if tiny" effects (common at n=500,000) from ones worth
    discussing in the text."""
    return abs(d) >= EFFECT_SIZE_NEGLIGIBLE
