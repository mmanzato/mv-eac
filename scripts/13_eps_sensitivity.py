#!/usr/bin/env python3
"""
Optional check -- sensitivity of the entity view to the KL smoothing constant epsilon.

Reranks the 50,000-impression validation subsample with Entity-EAC at each base
model's selected (lambda, beta) (read from best_params.csv, i.e. step 6) for every
epsilon in the grid, and records NDCG@K, ERR@K, and TCI@K. lambda and beta are NOT
re-selected per epsilon. The paper (Limitations) reports that NDCG@10 and TCI@10 are
stable, while ERR@10 for ItemKNN decreases monotonically as epsilon grows.

Usage:
    python scripts/13_eps_sensitivity.py [--models most_popular itemknn nrms]
                                         [--eps 1e-10 1e-6 1e-4 1e-2]

Output (under $MVEAC_DATA_ROOT/results/): eps_sensitivity.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.calibration.eac import rerank_all
from mveac.evaluation.metrics_runner import compute_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METHOD = "entity_eac"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-10, 1e-6, 1e-4, 1e-2])
    ap.add_argument("--n-impressions", type=int, default=None,
                    help="evaluate only the first N validation impressions (smoke test)")
    args = ap.parse_args()

    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        article_maps = pickle.load(f)
    with open(C.SEM_DIR / "category_vectors.pkl", "rb") as f:
        category_vectors = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        all_profiles = pickle.load(f)
    user_profiles = {v: all_profiles[v] for v in C.CANDIDATE_VIEWS}
    del all_profiles

    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    if args.n_impressions:
        val_sub = val_sub.head(args.n_impressions)
    best = pd.read_csv(C.RESULTS_DIR / "best_params.csv")

    rows = []
    for model in args.models:
        df = pd.read_parquet(C.SCORES_DIR / f"scores_{model}_val.parquet")
        df = df[df["impression_id"].isin(val_sub["impression_id"])]
        scores = {
            int(iid): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
            for iid, g in df.groupby("impression_id")
        }
        sel = best[(best["model"] == model) & (best["method"] == METHOD)].iloc[0]
        lam, beta = float(sel["eac_lambda"]), float(sel["eac_beta"])
        for eps in args.eps:
            t0 = time.time()
            ranked = rerank_all(scores, val_sub, user_profiles, article_maps, vocabs,
                                method=METHOD, lam=lam, beta=beta, k=C.K, epsilon=eps)
            metrics = compute_all(ranked, val_sub, article_maps, vocabs, category_vectors, C.K)
            rows.append({"model": model, "eps": eps, "lambda": lam, "beta": beta,
                         **{m: metrics[m] for m in ("NDCG@K", "ERR@K", "TCI@K")}})
            log.info("%s eps=%.0e NDCG=%.4f ERR=%.4f TCI=%.4f [%.0fs]", model, eps,
                     metrics["NDCG@K"], metrics["ERR@K"], metrics["TCI@K"], time.time() - t0)

    out = C.RESULTS_DIR / "eps_sensitivity.csv"
    if args.n_impressions:
        out = C.RESULTS_DIR / "eps_sensitivity_smoke.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("-> %s", out)


if __name__ == "__main__":
    main()
