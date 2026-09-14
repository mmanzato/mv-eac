"""
MAUT (Multi-Attribute Utility Theory) hyperparameter selection.

A single evaluation metric is insufficient to select (lambda, beta) for a
system that targets several objectives simultaneously: optimizing purely for
NDCG@K would push lambda to 0 (no calibration at all), and optimizing purely
for ERR@K/TCI@K would favor aggressive reranking at a large accuracy cost.
MAUT resolves this by linearly normalizing each metric to [0, 1] across the
validation grid, inverting the metrics that are better when *lower*
(ERR@K, TCI@K), and taking a weighted sum -- the single (lambda, beta) that
maximizes this composite score is the one reported for that method and base
model.
"""
from __future__ import annotations

import pandas as pd

from mveac.config import BETA_CAP, MAUT_EQUAL_WEIGHTS, MAUT_MINIMIZE, PARSIMONY_TOL


def maut_score(grid: pd.DataFrame, weights: dict[str, float] = None) -> pd.Series:
    """Composite MAUT utility for every row of a validation grid.

    ``grid`` must have one column per key of ``weights`` (default:
    ``mveac.config.MAUT_EQUAL_WEIGHTS``, i.e. NDCG@K/Entropy/ERR@K/TCI@K at
    0.25 each). Each column is min-max normalized *within this grid* before
    weighting, so the resulting score is only meaningful for comparing rows
    of the same grid against each other (e.g. across (lambda, beta) for one
    method/model), not across different grids.
    """
    weights = weights or MAUT_EQUAL_WEIGHTS
    score = pd.Series(0.0, index=grid.index)
    for metric, weight in weights.items():
        column = grid[metric].astype(float)
        lo, hi = column.min(), column.max()
        normalized = pd.Series(0.5, index=grid.index) if hi == lo else (column - lo) / (hi - lo)
        if metric in MAUT_MINIMIZE:
            normalized = 1.0 - normalized
        score = score + weight * normalized
    return score


def select_best_eac(grid: pd.DataFrame, score_column: str = "maut_score") -> pd.Series:
    """Pick the (lambda, beta) row EAC/MV-EAC reports for one method/model.

    Two refinements on top of a plain argmax, both documented in the paper
    (Section 4.6):

    1. **Selection grid restricted to beta <= BETA_CAP (0.9).** The validation
       grid is also evaluated at beta in {1.0, 1.5, 2.0}, but only as a
       sensitivity analysis: those points are NOT eligible for selection
       (they still enter the min-max normalization of the MAUT score). The
       extension does not show a plateau -- without the restriction, 9 of the
       15 EAC selections would move to beta > 0.9 (paper Section 5.2 and
       ``select_unrestricted`` below).
    2. **Parsimony tie-break.** Within the capped grid, any row within
       ``PARSIMONY_TOL`` of the best score is treated as tied with it; among
       those, the smallest (beta, lambda) is preferred, so the reported
       operating point is the least aggressive one that is not
       measurably worse than the best.
    """
    capped = grid[grid["beta"] <= BETA_CAP]
    best = capped[score_column].max()
    tied = capped[capped[score_column] >= best - PARSIMONY_TOL]
    return tied.sort_values(["beta", "lambda"]).iloc[0]


def select_unrestricted(grid: pd.DataFrame, score_column: str = "maut_score") -> pd.Series:
    """Same rule as ``select_best_eac`` but with every beta eligible -- used only for the
    sensitivity table that shows how the selection would change without the cap."""
    best = grid[score_column].max()
    tied = grid[grid[score_column] >= best - PARSIMONY_TOL]
    return tied.sort_values(["beta", "lambda"]).iloc[0]


def select_best_traditional(grid: pd.DataFrame, score_column: str = "maut_score") -> pd.Series:
    """Trad-Cal's lambda is selected independently of its paired EAC method: the MAUT
    score (min-max normalized over the full grid, as for EAC) is maximized over the
    beta=0 slice of the same grid only, so Trad-Cal is evaluated at its
    own best deterministic configuration rather than inheriting a lambda tuned jointly
    with an exploration term it does not use."""
    traditional_slice = grid[grid["beta"] == 0.0]
    return traditional_slice.loc[traditional_slice[score_column].idxmax()]
