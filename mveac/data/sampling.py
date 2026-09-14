"""
Stratified subsampling of the EB-NeRD Large validation behaviors.

EB-NeRD Large ships train/validation splits but no held-out test partition, so
this project draws two *disjoint* stratified subsamples from the validation
behaviors: a 50,000-impression subsample used only for hyperparameter
selection (Section 4.6 of the paper), and a 500,000-impression subsample
reserved for the final reported evaluation (Section 5). Both are stratified
over the same two-way grid so neither subsample over- or under-represents a
particular kind of user or a particular time window.

Stratification axes:
    1. user history-length quartile  (q_hist) -- cold vs. heavy users
    2. impression-timestamp quartile (q_ts)   -- early vs. late in the two-week window

giving 4 x 4 = 16 strata. Allocation is EQUAL across strata (n_total // 16 per
stratum, remainder to the first strata), not proportional to stratum size.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _assign_strata(
    behaviors: pd.DataFrame,
    history_sizes: dict[int, int],
    n_quantiles: int,
) -> pd.DataFrame:
    """Attach ``q_hist``, ``q_ts`` and a combined ``stratum`` column."""
    df = behaviors.copy()
    df["hist_size"] = df["user_id"].map(history_sizes).fillna(0).astype(int)
    df["ts"] = pd.to_datetime(df["impression_time"]).astype(np.int64) // 10**9

    df["q_hist"] = pd.qcut(df["hist_size"], n_quantiles, labels=False, duplicates="drop")
    df["q_ts"] = pd.qcut(df["ts"], n_quantiles, labels=False, duplicates="drop")
    df["stratum"] = df["q_hist"].astype(int) * n_quantiles + df["q_ts"].astype(int)
    return df


def _sample_proportionally(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Draw ``n_total`` rows from ``df`` with EQUAL allocation across its ``stratum`` values.

    (The historical name is kept for backward compatibility; the allocation is
    not proportional to stratum size -- this reproduces the paper's subsamples.)"""
    rng = np.random.default_rng(seed)
    strata = df["stratum"].value_counts().sort_index()
    n_strata = len(strata)
    per_stratum = n_total // n_strata
    leftover = n_total - per_stratum * n_strata

    parts = []
    for i, stratum in enumerate(strata.index):
        chunk = df[df["stratum"] == stratum]
        n_take = per_stratum + (1 if i < leftover else 0)
        n_take = min(n_take, len(chunk))
        parts.append(chunk.sample(n=n_take, random_state=int(rng.integers(1_000_000_000))))

    result = pd.concat(parts, ignore_index=True)
    log.info("Stratified sample: %d impressions across %d strata", len(result), n_strata)
    return result


def build_val_test_subsamples(
    validation_behaviors: pd.DataFrame,
    history_sizes: dict[int, int],
    val_n: int,
    test_n: int,
    n_quantiles: int,
    val_seed: int,
    test_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the disjoint (val, test) stratified subsamples used throughout the paper.

    ``history_sizes`` maps ``user_id -> number of articles in that user's
    training-window reading history`` (see ``mveac.data.loader.build_user_history_map``);
    users absent from the map are treated as having history length 0 (cold-start).
    """
    strat = _assign_strata(validation_behaviors, history_sizes, n_quantiles)

    val_sub = _sample_proportionally(strat, val_n, seed=val_seed)
    val_ids = set(val_sub["impression_id"])

    pool = strat[~strat["impression_id"].isin(val_ids)]
    test_sub = _sample_proportionally(pool, test_n, seed=test_seed)

    drop_cols = ["hist_size", "ts", "q_hist", "q_ts", "stratum"]
    val_sub = val_sub.drop(columns=[c for c in drop_cols if c in val_sub.columns])
    test_sub = test_sub.drop(columns=[c for c in drop_cols if c in test_sub.columns])

    log.info(
        "val=%d impressions (%d users) | test=%d impressions (%d users)",
        len(val_sub), val_sub["user_id"].nunique(),
        len(test_sub), test_sub["user_id"].nunique(),
    )
    return val_sub, test_sub


def candidate_set_size_report(behaviors: pd.DataFrame, k: int) -> dict[str, float]:
    """Summarize the candidate-set size |R_u| distribution (paper Section 4.1, Table 1).

    Impressions with ``|R_u| <= k`` leave every set-level (order-independent)
    metric -- ILD, Entropy, ERR, TCI -- structurally identical across every
    compared method, since the top-K list *is* the full candidate set
    regardless of how it is reranked; only NDCG can still differ on those
    impressions. This report quantifies how large that subset is.
    """
    sizes = behaviors["article_ids_inview"].apply(len)
    n_le_k = int((sizes <= k).sum())
    n_gt_k = int((sizes > k).sum())
    gt = sizes[sizes > k]
    report = {
        "n_impressions": int(len(sizes)),
        "median": float(sizes.median()),
        "mean": float(sizes.mean()),
        "p25": float(sizes.quantile(0.25)),
        "p75": float(sizes.quantile(0.75)),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "n_le_k": n_le_k,
        "n_gt_k": n_gt_k,
        "frac_le_k": n_le_k / len(sizes) if len(sizes) else 0.0,
        "gt_k_median": float(gt.median()) if len(gt) else float("nan"),
        "gt_k_p25": float(gt.quantile(0.25)) if len(gt) else float("nan"),
        "gt_k_p75": float(gt.quantile(0.75)) if len(gt) else float("nan"),
        "gt_k_mean_discarded": float((gt - k).mean()) if len(gt) else float("nan"),
        "n_eq_k_plus_1": int((sizes == k + 1).sum()),
    }
    if "user_id" in behaviors.columns:
        report["n_users"] = int(behaviors["user_id"].nunique())
    if "article_ids_clicked" in behaviors.columns:
        report["frac_with_click"] = float((behaviors["article_ids_clicked"].apply(len) > 0).mean())
    return report
