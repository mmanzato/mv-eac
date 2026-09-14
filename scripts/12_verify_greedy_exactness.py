#!/usr/bin/env python3
"""
Optional check -- is greedy Trad-Cal already the exact optimum for single-label views?

For a random sample of test impressions, reranks each impression with the greedy
Trad-Cal reranker and with the exact MCF solver at the same lambda, and compares the
value of the set-selection objective

    (1 - lambda) * sum_{i in S} Rel(i)  -  lambda * KL(P_u || Q_S),   Q_S = counts / |S|

For single-label views (category, sentiment) the objective is a modular relevance term
plus separable concave functions of the per-category counts, for which greedy selection
under a cardinality constraint is optimal; the paper (Section 4.4) reports that MCF never
finds a better set and that differing sets always tie in objective value.

Usage:
    python scripts/12_verify_greedy_exactness.py [--n-impressions 3000] [--lambdas 0.1 0.7 0.9]

Output (under $MVEAC_DATA_ROOT/results/): greedy_vs_mcf.csv
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
from mveac.calibration.eac import traditional_rerank
from mveac.calibration.mcf import mcf_rerank
from mveac.calibration.traditional import minmax_normalize


def objective(selected, candidates, profile, article_map, vocab, lam, eps):
    rel = dict(zip([a for a, _ in candidates], minmax_normalize([s for _, s in candidates])))
    index = {v: i for i, v in enumerate(vocab)}
    q = np.zeros(len(vocab))
    for aid in selected:
        for v in article_map.get(aid, ()):
            if v in index:
                q[index[v]] += 1
    q /= len(selected)
    return (1 - lam) * sum(rel[a] for a in selected) - lam * np.sum(profile * np.log((profile + eps) / (q + eps)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-impressions", type=int, default=3000)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.7, 0.9])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    article_maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
    all_profiles = pickle.load(open(C.SEM_DIR / "user_profiles.pkl", "rb"))
    views = ["category", "sentiment"]
    profiles = {v: all_profiles[v] for v in views}
    del all_profiles
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    imp2user = dict(zip(test_sub.impression_id.astype(int), test_sub.user_id.astype(int)))
    rng = np.random.default_rng(args.seed)
    sample = set(int(x) for x in rng.choice(test_sub.impression_id.values, args.n_impressions, replace=False))

    rows = []
    for model in C.MODELS:
        fname = "scores_nrms_seed0_test.parquet" if model == "nrms" else f"scores_{model}_test.parquet"
        df = pd.read_parquet(C.SCORES_DIR / fname)
        df = df[df.impression_id.isin(sample)]
        cands = {int(i): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
                 for i, g in df.groupby("impression_id")}
        for view in views:
            for lam in args.lambdas:
                n_diff = n_tie = n_mcf_better = n_greedy_better = 0
                for iid, cand in cands.items():
                    p = profiles[view].get(imp2user[iid])
                    p = np.ones(len(vocabs[view])) / len(vocabs[view]) if p is None else np.asarray(p, dtype=np.float64)
                    greedy = [a for a, _ in traditional_rerank(cand, p, article_maps[view], vocabs[view], lam, C.K, C.KL_EPSILON)]
                    exact = [a for a, _ in mcf_rerank(cand, p, article_maps[view], vocabs[view], lam, C.K)]
                    if set(greedy) == set(exact):
                        continue
                    n_diff += 1
                    gap = (objective(exact, cand, p, article_maps[view], vocabs[view], lam, C.KL_EPSILON)
                           - objective(greedy, cand, p, article_maps[view], vocabs[view], lam, C.KL_EPSILON))
                    if abs(gap) < 1e-5:
                        n_tie += 1
                    elif gap > 0:
                        n_mcf_better += 1
                    else:
                        n_greedy_better += 1
                rows.append({"model": model, "view": view, "lambda": lam, "n_impressions": len(cands),
                             "different_sets": n_diff, "different_sets_equal_objective": n_tie,
                             "mcf_better": n_mcf_better, "greedy_better": n_greedy_better})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "greedy_vs_mcf.csv", index=False)


if __name__ == "__main__":
    main()
