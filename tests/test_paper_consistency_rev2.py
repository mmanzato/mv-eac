"""Consistency of the shipped second-revision results (data/results/rev2) with numbers stated in
the revised paper. These tests read reference outputs only; they do not rerun the pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R = Path(__file__).resolve().parent.parent / "data" / "results" / "rev2"
pytestmark = pytest.mark.skipif(not R.exists(), reason="data/results/rev2 not present")


def _sel(config):
    b = pd.read_csv(R / "best_params_rev2.csv")
    return {m: (r["lambda"], r.beta) for m, r in b[b.config == config].set_index("model").iterrows()}


def test_selected_hyperparameters():
    assert _sel("mv_eac_norm") == {"most_popular": (0.9, 5.0), "itemknn": (0.9, 2.0), "nrms": (0.3, 1.0)}
    assert _sel("entity_eac") == {"most_popular": (0.9, 10.0), "itemknn": (0.1, 20.0), "nrms": (0.1, 10.0)}
    assert _sel("tradcal_mv_norm") == {"most_popular": (0.9, 0.0), "itemknn": (0.9, 0.0), "nrms": (0.5, 0.0)}


def test_no_selection_at_the_top_of_its_grid():
    b = pd.read_csv(R / "best_params_rev2.csv")
    eac = b[b.config.isin(["category_eac", "topic_eac", "sentiment_eac", "mv_eac", "mv_eac_norm"])]
    assert (eac.beta < 10).all()
    assert (b[b.config == "entity_eac"].beta < 100).all()


def test_main_family_counts():
    s = pd.read_csv(R / "stats" / "stats_rev2_full.csv")
    assert len(s) == 735
    assert s.sig_uncorr.sum() == 712 and s.sig_holm.sum() == 702 and s.sig_bh.sum() == 712
    lost = s[s.sig_uncorr & ~s.sig_holm]
    assert len(lost) == 10 and lost.d_imp.abs().max() <= 0.014 + 1e-9
    assert (s.sig_holm != s.sig_user_holm).sum() == 10
    assert set(s.paper_protocol) == set("ABCDEFGHIJKLMNOP")


def test_headline_effects():
    s = pd.read_csv(R / "stats" / "stats_rev2_full.csv")
    a = s[(s.comparison == "mv_eac_norm vs tradcal_mv_norm") & (s.family == "A_EAC_vs_TradCal")].set_index(["model", "metric"])
    tci = a.xs("TCI@K", level="metric").d_imp
    ent = a.xs("Entropy", level="metric").d_imp
    assert tci.round(2).between(-0.37, -0.34).all()
    assert ent.round(2).between(0.42, 0.44).all()


def test_secondary_and_mind_families():
    sec = pd.read_csv(R / "stats" / "stats_rev2_secondary.csv")
    assert len(sec) == 1728 and sec.sig_holm.sum() == 1693
    mind = pd.read_csv(R / "mind" / "stats_mind_full.csv")
    assert len(mind) == 100 and mind.sig_holm.sum() == 94


def test_normalization_adopted_in_every_model():
    v = pd.read_csv(R / "val_compare_norm.csv")
    for _, g in v.groupby("model"):
        assert not (g.verdict == "Norm WORSE").any() and (g.verdict == "Norm better").any()


def test_base_model_accuracy():
    a = pd.read_csv(R / "base_model_accuracy.csv")
    a = a[a.subset == "all"].set_index("model")
    assert np.allclose(a.AUC.round(3)[["most_popular", "itemknn", "nrms"]].values, [0.426, 0.507, 0.598])
