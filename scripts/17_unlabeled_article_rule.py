#!/usr/bin/env python3
"""
Optional analysis -- robustness to the handling of articles with no label under a view
(Supplementary Section S7).

The paper's rule gives a candidate with no label under a view no exploration bonus for that
view (UCB = 0). This script reranks the validation subsample with Entity-EAC and MV-EAC at
their selected (lambda, beta) under that rule ("zero") and under the alternative rule that
gives such candidates the largest bonus Eq. (3) allows ("max"), and compares the two.

Usage:
    python scripts/17_unlabeled_article_rule.py [--workers 6]

Output (under $MVEAC_DATA_ROOT/results/): unlabeled_article_rule.csv
    one row per (model, method, metric): mean under each rule, paired difference (max - zero),
    Cohen's d of that difference, and the share of impressions whose list changes
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.analysis.greedy_diagnostics import diagnostic_multi_view_eac
from mveac.evaluation.metrics_runner import per_impression
from mveac.evaluation.stats import cohens_d

METRICS = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
_SHARED: dict = {}


def _run(task):
    model, method, rule = task
    views = C.MV_VIEWS if method == "mv_eac" else ["entity"]
    weights = C.MV_UNIFORM_WEIGHTS if method == "mv_eac" else {"entity": 1.0}
    lam, beta = _SHARED["params"][(model, method)]
    vocabs, maps, profiles = _SHARED["vocabs"], _SHARED["maps"], _SHARED["profiles"]
    ranked = {}
    for iid, candidates in _SHARED["scores"][model].items():
        uid = _SHARED["imp2user"][iid]
        prof = {v: profiles[v].get(uid, np.ones(len(vocabs[v])) / len(vocabs[v])) for v in views}
        ranked[iid] = diagnostic_multi_view_eac(candidates, prof, maps, vocabs, weights, lam, beta, C.K,
                                                C.KL_EPSILON, unlabeled_rule=rule)
    per = per_impression(ranked, _SHARED["val_sub"], maps, vocabs, _SHARED["catvec"], C.K)
    lists = {iid: tuple(a for a, _ in r) for iid, r in ranked.items()}
    return task, pd.DataFrame([{"impression_id": i, **m} for i, m in per.items()]).set_index("impression_id"), lists


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    args = ap.parse_args()

    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
    with open(C.SEM_DIR / "category_vectors.pkl", "rb") as f:
        catvec = pickle.load(f)
    all_profiles = pickle.load(open(C.SEM_DIR / "user_profiles.pkl", "rb"))
    profiles = {v: all_profiles[v] for v in C.MV_VIEWS}
    del all_profiles
    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    best = pd.read_csv(C.RESULTS_DIR / "best_params.csv")
    scores = {}
    for model in C.MODELS:
        df = pd.read_parquet(C.SCORES_DIR / f"scores_{model}_val.parquet")
        scores[model] = {int(i): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
                         for i, g in df.groupby("impression_id")}
    _SHARED.update(vocabs=vocabs, maps=maps, catvec=catvec, profiles=profiles, val_sub=val_sub, scores=scores,
                   imp2user=dict(zip(val_sub.impression_id.astype(int), val_sub.user_id.astype(int))),
                   params={(r.model, r.method): (float(r.eac_lambda), float(r.eac_beta)) for r in best.itertuples()})

    tasks = [(m, meth, rule) for m in C.MODELS for meth in ["entity_eac", "mv_eac"] for rule in ["zero", "max"]]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        results = {task: (df, lists) for task, df, lists in pool.map(_run, tasks)}

    rows = []
    for model in C.MODELS:
        for method in ["entity_eac", "mv_eac"]:
            (zero, zl), (maxr, ml) = results[(model, method, "zero")], results[(model, method, "max")]
            maxr = maxr.loc[zero.index]
            changed = float(np.mean([zl[i] != ml[i] for i in zero.index]))
            for metric in METRICS:
                diff = (maxr[metric] - zero[metric]).to_numpy()
                rows.append({"model": model, "method": method, "metric": metric,
                             "mean_zero_rule": zero[metric].mean(), "mean_max_rule": maxr[metric].mean(),
                             "delta": diff.mean(), "d": cohens_d(diff), "share_lists_changed": changed})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "unlabeled_article_rule.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
