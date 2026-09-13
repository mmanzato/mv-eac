#!/usr/bin/env python3
"""
Step 11 (optional) -- reranking latency and profile-construction cost.

Times ONLY the per-impression reranking call (single process, BLAS/OpenMP pinned to
1 thread) for every reranker at its selected hyperparameters, on a fixed random
sample of test impressions per base model, and the per-user cost of building the
semantic profiles from EB-NeRD history for a random sample of test users.

Usage:
    python scripts/11_latency_benchmark.py [--n-impressions 2000] [--n-users 2000]

Output (under $MVEAC_DATA_ROOT/results/): latency_bench.csv, profile_cost_bench.csv
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.calibration.eac import multi_view_eac, single_view_eac, traditional_rerank
from mveac.calibration.mcf import mcf_rerank
from mveac.data.loader import load_train_history
from mveac.data.semantic_profiles import _normalized_frequency as build_profile

VIEWS = ["category", "entity", "topic", "sentiment"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-impressions", type=int, default=2000)
    ap.add_argument("--n-users", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    article_maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
    all_profiles = pickle.load(open(C.SEM_DIR / "user_profiles.pkl", "rb"))
    profiles = {v: all_profiles[v] for v in VIEWS}
    del all_profiles
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    imp2user = dict(zip(test_sub.impression_id.astype(int), test_sub.user_id.astype(int)))
    best = pd.read_csv(C.RESULTS_DIR / "best_params.csv")
    best_mcf_path = C.RESULTS_DIR / "best_params_mcf.csv"
    best_mcf = pd.read_csv(best_mcf_path) if best_mcf_path.exists() else None
    rng = np.random.default_rng(args.seed)
    sample = set(int(x) for x in rng.choice(test_sub.impression_id.astype(int).values,
                                            size=args.n_impressions, replace=False))

    def profile(view, uid):
        p = profiles[view].get(uid)
        return np.ones(len(vocabs[view])) / len(vocabs[view]) if p is None else np.asarray(p, dtype=np.float64)

    rows = []
    for model in C.MODELS:
        fname = "scores_nrms_seed0_test.parquet" if model == "nrms" else f"scores_{model}_test.parquet"
        df = pd.read_parquet(C.SCORES_DIR / fname)
        df = df[df.impression_id.isin(sample)]
        cands = {int(i): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
                 for i, g in df.groupby("impression_id")}
        bp = lambda m, col: float(best[(best.model == model) & (best.method == m)][col].iloc[0])

        def timed(fn):
            ts = []
            for iid, cand in cands.items():
                t0 = time.perf_counter(); fn(cand, imp2user[iid]); ts.append((time.perf_counter() - t0) * 1e3)
            return np.array(ts)

        runners = {"original": lambda cand, uid: cand[:C.K]}
        for v in VIEWS:
            lam, beta, tl = bp(f"{v}_eac", "eac_lambda"), bp(f"{v}_eac", "eac_beta"), bp(f"{v}_eac", "trad_lambda")
            runners[f"traditional_{v}"] = (lambda cand, uid, v=v, tl=tl: traditional_rerank(
                cand, profile(v, uid), article_maps[v], vocabs[v], tl, C.K, C.KL_EPSILON))
            runners[f"{v}_eac"] = (lambda cand, uid, v=v, lam=lam, beta=beta: single_view_eac(
                cand, profile(v, uid), article_maps[v], vocabs[v], lam, beta, C.K, C.KL_EPSILON))
            if best_mcf is not None:
                ml = float(best_mcf[(best_mcf.model == model) & (best_mcf.view == v)]["lambda"].iloc[0])
                runners[f"mcf_{v}"] = (lambda cand, uid, v=v, ml=ml: mcf_rerank(
                    cand, profile(v, uid), article_maps[v], vocabs[v], ml, C.K))
        for name, lam, beta in [("traditional_mv", bp("mv_eac", "trad_lambda"), 0.0),
                                ("mv_eac", bp("mv_eac", "eac_lambda"), bp("mv_eac", "eac_beta"))]:
            runners[name] = (lambda cand, uid, lam=lam, beta=beta: multi_view_eac(
                cand, {v: profile(v, uid) for v in C.MV_VIEWS}, article_maps, vocabs,
                C.MV_UNIFORM_WEIGHTS, lam, beta, C.K, C.KL_EPSILON))
        for name, fn in runners.items():
            ts = timed(fn)
            rows.append({"model": model, "method": name, "n_impressions": len(ts), "median_ms": np.median(ts),
                         "mean_ms": ts.mean(), "p95_ms": np.percentile(ts, 95)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "latency_bench.csv", index=False)

    history = load_train_history(C.EBNERD_LARGE_ZIP)
    users = set(int(u) for u in rng.choice(test_sub.user_id.astype(int).unique(), size=args.n_users, replace=False))
    history = history[history.user_id.isin(users)]
    hmap = {int(r.user_id): [int(x) for x in r.article_id_fixed] for r in history.itertuples(index=False)}
    prow = []
    for v in VIEWS:
        ts = []
        for h in hmap.values():
            t0 = time.perf_counter(); build_profile(h, article_maps[v], vocabs[v]); ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        prow.append({"view": v, "n_users": len(ts), "median_history_len": float(np.median([len(h) for h in hmap.values()])),
                     "median_ms": np.median(ts), "mean_ms": ts.mean(), "p95_ms": np.percentile(ts, 95)})
        print(prow[-1], flush=True)
    pd.DataFrame(prow).to_csv(C.RESULTS_DIR / "profile_cost_bench.csv", index=False)


if __name__ == "__main__":
    main()
