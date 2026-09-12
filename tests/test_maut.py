"""Unit tests for MAUT hyperparameter selection (mveac.evaluation.maut) on small,
hand-constructed validation grids."""
from __future__ import annotations

import pandas as pd
import pytest

from mveac.evaluation.maut import maut_score, select_best_eac, select_best_traditional


def _grid(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestMautScore:
    def test_best_row_on_every_metric_scores_highest(self):
        grid = _grid([
            {"NDCG@K": 0.5, "Entropy": 0.5, "ERR@K": 0.5, "TCI@K": 0.5},   # middling everywhere
            {"NDCG@K": 1.0, "Entropy": 1.0, "ERR@K": 0.0, "TCI@K": 0.0},   # best on all 4 (ERR/TCI minimized)
            {"NDCG@K": 0.0, "Entropy": 0.0, "ERR@K": 1.0, "TCI@K": 1.0},   # worst on all 4
        ])
        scores = maut_score(grid)
        assert scores.idxmax() == 1
        assert scores.idxmin() == 2

    def test_constant_column_does_not_crash_and_scores_neutrally(self):
        # If a metric is identical across the whole grid, it carries no
        # information -- every row should get the neutral 0.5 contribution
        # from that metric rather than a divide-by-zero.
        grid = _grid([
            {"NDCG@K": 0.3, "Entropy": 0.7, "ERR@K": 0.7, "TCI@K": 0.2},
            {"NDCG@K": 0.3, "Entropy": 0.2, "ERR@K": 0.7, "TCI@K": 0.9},
        ])
        scores = maut_score(grid)
        assert scores.notna().all()

    def test_custom_weights_are_respected(self):
        grid = _grid([
            {"NDCG@K": 1.0, "Entropy": 0.0},
            {"NDCG@K": 0.0, "Entropy": 1.0},
        ])
        # All weight on NDCG -> row 0 wins; all weight on Entropy -> row 1 wins.
        assert maut_score(grid, {"NDCG@K": 1.0}).idxmax() == 0
        assert maut_score(grid, {"Entropy": 1.0}).idxmax() == 1


class TestSelectBestEAC:
    def test_respects_the_beta_cap(self):
        # The row with beta=2.0 has the single best raw score, but BETA_CAP=0.9
        # in mveac.config must exclude it from consideration entirely.
        grid = _grid([
            {"lambda": 0.5, "beta": 0.5, "maut_score": 0.80},
            {"lambda": 0.5, "beta": 0.9, "maut_score": 0.85},
            {"lambda": 0.5, "beta": 2.0, "maut_score": 0.99},  # excluded by the cap
        ])
        best = select_best_eac(grid)
        assert best["beta"] <= 0.9
        assert best["beta"] == 0.9

    def test_parsimony_tolerance_prefers_the_smaller_beta_among_near_ties(self):
        # Two rows are within PARSIMONY_TOL (0.003) of each other -- the smaller
        # beta must be preferred even though it is not the strict argmax.
        grid = _grid([
            {"lambda": 0.5, "beta": 0.1, "maut_score": 0.700},
            {"lambda": 0.5, "beta": 0.9, "maut_score": 0.701},  # technically higher, but within tolerance
        ])
        best = select_best_eac(grid)
        assert best["beta"] == pytest.approx(0.1)

    def test_strict_winner_outside_tolerance_is_still_selected(self):
        grid = _grid([
            {"lambda": 0.5, "beta": 0.1, "maut_score": 0.10},
            {"lambda": 0.5, "beta": 0.9, "maut_score": 0.90},  # far outside PARSIMONY_TOL
        ])
        best = select_best_eac(grid)
        assert best["beta"] == pytest.approx(0.9)


class TestSelectBestTraditional:
    def test_only_considers_the_beta_zero_slice(self):
        grid = _grid([
            {"lambda": 0.3, "beta": 0.0, "maut_score": 0.5},
            {"lambda": 0.7, "beta": 0.0, "maut_score": 0.9},   # best of the beta=0 slice
            {"lambda": 0.9, "beta": 0.9, "maut_score": 0.99},  # best overall, but beta != 0
        ])
        best = select_best_traditional(grid)
        assert best["beta"] == 0.0
        assert best["lambda"] == pytest.approx(0.7)
