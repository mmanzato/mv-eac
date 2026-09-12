#!/usr/bin/env python3
"""
Step 5 of 10 -- validation-set grid search over (lambda, beta) for every EAC
method (single-view and MV-EAC).

For each (base model, method, lambda, beta) combination, reranks the
50,000-impression validation subsample and records the mean of every metric.
This grid is what step 6 (MAUT selection) reads to pick one (lambda, beta)
per method/model.

Runs one (model, method) combination at a time by default, in parallel
worker processes over the (lambda, beta) grid; resumable -- already-computed
rows are skipped on rerun. Fork-based multiprocessing is used since the
shared vocabularies/profiles/article maps (loaded once per worker via
``--workers``) are large enough that re-pickling them per task would be the
dominant cost.

Usage:
    python scripts/05_grid_search.py --models most_popular --methods mv_eac
    python scripts/05_grid_search.py   # all models x all methods (slow: budget hours)

Output (under $MVEAC_DATA_ROOT/results/):
    val_grid_{model}_{method}.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.calibration.eac import rerank_all
from mveac.evaluation.metrics_runner import compute_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SHARED: dict = {}


def _load_shared(models: list[str]) -> None:
    t0 = time.time()
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        article_maps = pickle.load(f)
    with open(C.SEM_DIR / "category_vectors.pkl", "rb") as f:
        category_vectors = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        all_profiles = pickle.load(f)
    # Grid search only reranks with the four candidate single-views + MV-EAC's
    # own 3-view set, never subcategory -- drop it here to save memory.
    user_profiles = {v: all_profiles[v] for v in C.CANDIDATE_VIEWS}
    del all_profiles

    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    scores = {}
    for model in models:
        df = pd.read_parquet(C.SCORES_DIR / f"scores_{model}_val.parquet")
        scores[model] = {
            int(iid): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
            for iid, g in df.groupby("impression_id")
        }

    _SHARED.update(vocabs=vocabs, article_maps=article_maps, category_vectors=category_vectors,
                    user_profiles=user_profiles, val_sub=val_sub, scores=scores)
    log.info("Shared state loaded in %.1fs (val impressions=%d)", time.time() - t0, len(val_sub))


def _run_one(task: tuple[str, str, float, float]) -> dict:
    model, method, lam, beta = task
    weights = C.MV_UNIFORM_WEIGHTS if method == "mv_eac" else None
    t0 = time.time()
    ranked = rerank_all(
        _SHARED["scores"][model], _SHARED["val_sub"], _SHARED["user_profiles"],
        _SHARED["article_maps"], _SHARED["vocabs"], method=method, lam=lam, beta=beta,
        k=C.K, weights=weights, epsilon=C.KL_EPSILON,
    )
    metrics = compute_all(
        ranked, _SHARED["val_sub"], _SHARED["article_maps"], _SHARED["vocabs"],
        _SHARED["category_vectors"], C.K,
    )
    return {"model": model, "method": method, "lambda": lam, "beta": beta,
            "_elapsed_s": round(time.time() - t0, 1), **metrics}


def _already_done(model: str, method: str) -> set[tuple[float, float]]:
    path = C.RESULTS_DIR / f"val_grid_{model}_{method}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["lambda"].round(3), df["beta"].round(4)))


def _append_row(row: dict) -> None:
    path = C.RESULTS_DIR / f"val_grid_{row['model']}_{row['method']}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    rows = [r for r in rows if not (round(r["lambda"], 3) == round(row["lambda"], 3)
                                     and round(r["beta"], 4) == round(row["beta"], 4))]
    rows.append(row)
    pd.DataFrame(rows).sort_values(["lambda", "beta"]).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--methods", nargs="+", default=C.GRID_METHODS)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    args = ap.parse_args()

    _load_shared(args.models)

    tasks = []
    for model in args.models:
        for method in args.methods:
            done = _already_done(model, method)
            for lam, beta in itertools.product(C.LAMBDA_GRID, C.BETA_GRID):
                if (round(lam, 3), round(beta, 4)) not in done:
                    tasks.append((model, method, lam, beta))
    log.info("Pending grid points: %d (workers=%d)", len(tasks), args.workers)
    if not tasks:
        log.info("Nothing to do.")
        return

    ctx = mp.get_context("fork")
    t_start, done_n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception:
                log.exception("Task %s failed", task)
                continue
            _append_row(row)
            done_n += 1
            rate = done_n / (time.time() - t_start)
            eta_min = (len(tasks) - done_n) / rate / 60 if rate > 0 else float("nan")
            log.info("[%d/%d] %s/%s lam=%.2f beta=%.2f NDCG=%.4f ERR=%.4f TCI=%.4f (%.0fs, ETA %.0f min)",
                      done_n, len(tasks), row["model"], row["method"], row["lambda"], row["beta"],
                      row["NDCG@K"], row["ERR@K"], row["TCI@K"], row["_elapsed_s"], eta_min)

    log.info("Grid search complete in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
