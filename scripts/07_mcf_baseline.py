#!/usr/bin/env python3
"""
Step 7 of 10 -- the MCF (Minimum-Cost-Flow) calibration baseline.

MCF has no exploration term, so its validation grid is 1-dimensional (lambda
only) and much cheaper than the EAC grid search. Requires ``ortools``
(``pip install ortools``).

Usage:
    python scripts/07_mcf_baseline.py grid
    python scripts/07_mcf_baseline.py select
    python scripts/07_mcf_baseline.py eval

Output (under $MVEAC_DATA_ROOT/results/):
    val_grid_mcf_{model}_{view}.csv
    best_params_mcf.csv
    per_impression_mcf_{model}_{view}.csv
    test_summary_mcf.csv
"""
from __future__ import annotations

import argparse
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
from mveac.calibration.mcf import rerank_all_mcf
from mveac.evaluation.maut import maut_score
from mveac.evaluation.metrics_runner import compute_all, per_impression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VIEWS = ["category", "topic", "entity", "sentiment"]  # all four single-view candidates
_SHARED: dict = {}


def _load_shared(models: list[str], split: str) -> None:
    t0 = time.time()
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        article_maps = pickle.load(f)
    with open(C.SEM_DIR / "category_vectors.pkl", "rb") as f:
        category_vectors = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        user_profiles = pickle.load(f)  # MCF uses each view independently, all 4 needed

    behaviors = pd.read_parquet(C.PROC_DIR / f"{split}_sub.parquet")
    scores = {}
    for model in models:
        df = pd.read_parquet(C.SCORES_DIR / f"scores_{model}_{split}.parquet")
        scores[model] = {
            int(iid): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
            for iid, g in df.groupby("impression_id")
        }
    _SHARED.update(vocabs=vocabs, article_maps=article_maps, category_vectors=category_vectors,
                    user_profiles=user_profiles, behaviors=behaviors, scores=scores)
    log.info("Shared state loaded in %.1fs (%s impressions=%d)", time.time() - t0, split, len(behaviors))


def _run_grid_task(task: tuple[str, str, float]) -> dict:
    model, view, lam = task
    t0 = time.time()
    ranked = rerank_all_mcf(_SHARED["scores"][model], _SHARED["behaviors"], _SHARED["user_profiles"],
                            _SHARED["article_maps"], _SHARED["vocabs"], view, lam, C.K)
    metrics = compute_all(ranked, _SHARED["behaviors"], _SHARED["article_maps"], _SHARED["vocabs"],
                          _SHARED["category_vectors"], C.K)
    return {"model": model, "view": view, "lambda": lam, "_elapsed_s": round(time.time() - t0, 1), **metrics}


def cmd_grid(models: list[str], workers: int) -> None:
    _load_shared(models, "val")
    tasks = []
    for model in models:
        for view in VIEWS:
            path = C.RESULTS_DIR / f"val_grid_mcf_{model}_{view}.csv"
            done = set(pd.read_csv(path)["lambda"].round(3)) if path.exists() else set()
            tasks += [(model, view, lam) for lam in C.LAMBDA_GRID if round(lam, 3) not in done]
    log.info("Pending MCF grid points: %d", len(tasks))
    if not tasks:
        return
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_grid_task, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            path = C.RESULTS_DIR / f"val_grid_mcf_{row['model']}_{row['view']}.csv"
            rows = pd.read_csv(path).to_dict("records") if path.exists() else []
            rows = [r for r in rows if round(r["lambda"], 3) != round(row["lambda"], 3)]
            rows.append(row)
            pd.DataFrame(rows).sort_values("lambda").to_csv(path, index=False)
            log.info("[%d/%d] %s/mcf_%s lam=%.2f NDCG=%.4f ERR=%.4f TCI=%.4f",
                      i, len(tasks), row["model"], row["view"], row["lambda"],
                      row["NDCG@K"], row["ERR@K"], row["TCI@K"])


def cmd_select(models: list[str]) -> None:
    rows = []
    for model in models:
        for view in VIEWS:
            path = C.RESULTS_DIR / f"val_grid_mcf_{model}_{view}.csv"
            if not path.exists():
                log.warning("missing %s", path)
                continue
            grid = pd.read_csv(path)
            grid["maut_score"] = maut_score(grid)
            best = grid.loc[grid["maut_score"].idxmax()]
            rows.append({"model": model, "view": view, "lambda": float(best["lambda"])})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "best_params_mcf.csv", index=False)
    log.info("best_params_mcf.csv written (%d rows)", len(rows))


def _run_eval_task(task: tuple[str, str, float]) -> dict:
    model, view, lam = task
    t0 = time.time()
    ranked = rerank_all_mcf(_SHARED["scores"][model], _SHARED["behaviors"], _SHARED["user_profiles"],
                            _SHARED["article_maps"], _SHARED["vocabs"], view, lam, C.K)
    per = per_impression(ranked, _SHARED["behaviors"], _SHARED["article_maps"], _SHARED["vocabs"],
                         _SHARED["category_vectors"], C.K)
    df = pd.DataFrame([{"impression_id": k, **v} for k, v in per.items()])
    df.to_csv(C.RESULTS_DIR / f"per_impression_mcf_{model}_{view}.csv", index=False)
    agg = {"model": model, "method": f"mcf_{view}", "lambda": lam, "elapsed_s": round(time.time() - t0, 1)}
    for col in ("NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"):
        if col in df.columns:
            agg[col] = float(df[col].mean())
    return agg


def cmd_eval(models: list[str], workers: int) -> None:
    best = pd.read_csv(C.RESULTS_DIR / "best_params_mcf.csv")
    _load_shared(models, "test")
    tasks = []
    for model in models:
        for view in VIEWS:
            if (C.RESULTS_DIR / f"per_impression_mcf_{model}_{view}.csv").exists():
                continue
            row = best[(best.model == model) & (best.view == view)]
            lam = float(row["lambda"].iloc[0]) if len(row) else 0.5
            tasks.append((model, view, lam))
    log.info("Pending MCF eval configs: %d", len(tasks))
    if not tasks:
        return
    summary_path = C.RESULTS_DIR / "test_summary_mcf.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_eval_task, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            rows = [x for x in rows if (x["model"], x["method"]) != (r["model"], r["method"])]
            rows.append(r)
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            log.info("[%d/%d] %s/%s NDCG=%.4f ERR=%.4f TCI=%.4f",
                      i, len(tasks), r["model"], r["method"], r["NDCG@K"], r["ERR@K"], r["TCI@K"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["grid", "select", "eval"])
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    args = ap.parse_args()
    {"grid": lambda: cmd_grid(args.models, args.workers),
     "select": lambda: cmd_select(args.models),
     "eval": lambda: cmd_eval(args.models, args.workers)}[args.cmd]()
