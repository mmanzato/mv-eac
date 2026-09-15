#!/usr/bin/env python3
"""
Optional analysis -- per-view scale of the MV-EAC score terms (Supplementary Section S7).

The per-view KL and UCB terms of MV-EAC are not normalized to a common scale, so equal view
weights are nominal. Along MV-EAC's greedy path (selected lambda, beta; validation set), for a
random sample of impressions with more than K candidates per base model, records for every
step the spread across the remaining candidates of each weighted term, the mean unweighted KL
and UCB of each view, and whether dropping a view's KL and UCB terms would change the selected
article.

Usage:
    python scripts/16_view_term_scales.py [--n-impressions 3000] [--seed 0]

Output (under $MVEAC_DATA_ROOT/results/):
    view_term_scales_steps.csv    one row per (model, impression, greedy step)
    view_term_scales.csv          per model: median spreads, flip rates, median mean KL/UCB
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.analysis.greedy_diagnostics import diagnostic_multi_view_eac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-impressions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    article_maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
    all_profiles = pickle.load(open(C.SEM_DIR / "user_profiles.pkl", "rb"))
    profiles = {v: all_profiles[v] for v in C.MV_VIEWS}
    del all_profiles
    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    imp2user = dict(zip(val_sub.impression_id.astype(int), val_sub.user_id.astype(int)))
    best = pd.read_csv(C.RESULTS_DIR / "best_params.csv")
    rng = np.random.default_rng(args.seed)

    rows = []
    for model in C.MODELS:
        sel = best[(best.model == model) & (best.method == "mv_eac")].iloc[0]
        lam, beta = float(sel.eac_lambda), float(sel.eac_beta)
        scores = pd.read_parquet(C.SCORES_DIR / f"scores_{model}_val.parquet")
        sizes = scores.groupby("impression_id").size()
        sample = rng.choice(sizes[sizes > C.K].index.values, args.n_impressions, replace=False)
        scores = scores[scores.impression_id.isin(sample)]
        for iid, g in scores.groupby("impression_id"):
            candidates = [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
            uid = imp2user[int(iid)]
            prof = {v: profiles[v].get(uid, np.ones(len(vocabs[v])) / len(vocabs[v])) for v in C.MV_VIEWS}
            _, steps = diagnostic_multi_view_eac(candidates, prof, article_maps, vocabs, C.MV_UNIFORM_WEIGHTS,
                                                 lam, beta, C.K, C.KL_EPSILON, record_steps=True)
            rows += [{"model": model, "impression_id": int(iid), **s} for s in steps]

    steps = pd.DataFrame(rows)
    steps.to_csv(C.RESULTS_DIR / "view_term_scales_steps.csv", index=False)
    summary = pd.concat([
        steps.groupby("model")[[c for c in steps if c.startswith("sd_")]].median(),
        steps.groupby("model")[[c for c in steps if c.startswith("flip_")]].mean(),
        steps.groupby("model")[[c for c in steps if c.startswith("mean_")]].median(),
    ], axis=1)
    summary.to_csv(C.RESULTS_DIR / "view_term_scales.csv")
    print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
