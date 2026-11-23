#!/usr/bin/env python3
"""
Second revision, step 28 -- reranking latency of the normalized MV-EAC and of MV-EAC_nom (paper
Section 6.2, Supplementary Section S7). Same protocol as step 11: one process, one BLAS thread,
2,000 random test impressions (seed 0), timing only the per-impression rerank call at the selected
(lambda, beta), after 20 warm-up calls.

Output: results/rev2/latency_rev2.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.calibration.eac import eac_rerank

OUT = C.RESULTS_DIR / "rev2"
V3 = list(C.MV_VIEWS)


def main() -> None:
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        maps = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        pall = pickle.load(f)
    profiles = {v: pall[v] for v in V3}
    del pall
    sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    i2u = dict(zip(sub.impression_id.astype(int), sub.user_id.astype(int)))
    ids = set(int(x) for x in np.random.default_rng(0).choice(sub.impression_id.astype(int).values, 2000, replace=False))
    bp = pd.read_csv(OUT / "best_params_rev2.csv")

    def prof(v, u):
        p = profiles[v].get(u)
        return np.ones(len(vocabs[v])) / len(vocabs[v]) if p is None else np.asarray(p, dtype=np.float64)

    rows = []
    for model in C.MODELS:
        df = pd.read_parquet(C.score_path(model, "test"))
        df = df[df.impression_id.isin(ids)]
        cands = {int(i): [(int(a), float(s)) for a, s in zip(g.article_id, g.score)]
                 for i, g in df.sort_values(["impression_id", "rank"]).groupby("impression_id")}
        for meth, norm in [("mv_eac_norm", True), ("mv_eac", False)]:
            lam, beta = bp[(bp.model == model) & (bp.config == meth)][["lambda", "beta"]].iloc[0].astype(float)
            for iid in list(cands)[:20]:
                eac_rerank(cands[iid], {v: prof(v, i2u[iid]) for v in V3}, maps, vocabs, C.MV_UNIFORM_WEIGHTS,
                           lam, beta, C.K, normalize=norm)
            ts, nc = [], []
            for iid, c in cands.items():
                ps = {v: prof(v, i2u[iid]) for v in V3}
                t0 = time.perf_counter()
                eac_rerank(c, ps, maps, vocabs, C.MV_UNIFORM_WEIGHTS, lam, beta, C.K, normalize=norm)
                ts.append((time.perf_counter() - t0) * 1e3)
                nc.append(len(c))
            ts, nc = np.array(ts), np.array(nc)
            big = ts[nc > C.K]
            rows.append({"model": model, "method": meth, "median_ms": np.median(ts), "p95_ms": np.percentile(ts, 95),
                         "median_ms_gtK": np.median(big), "p95_ms_gtK": np.percentile(big, 95)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "latency_rev2.csv", index=False)


if __name__ == "__main__":
    main()
