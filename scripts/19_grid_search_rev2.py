#!/usr/bin/env python3
"""
Second revision, step 19 -- validation grids (50,000 impressions).

Tasks (run ``--task eac norm baselines`` first, then step 20 ``--stage a``, then
``--task views d1 simplex components``, then step 20 ``--stage b``):

  eac         single-view EAC (category, entity, topic, sentiment) and the un-normalized
              MV-EAC_nom over lambda x beta, beta in BETA_GRID_REV2 (up to 10, all eligible),
              plus Entity-EAC at beta in {20, 50, 100, inf} (paper Section 4.6)
  norm        the paper's MV-EAC (per-view normalization, Eq. 10) over lambda x beta
  baselines   MMR, xQuAD, DPP, Dedup over their one-parameter grids (stored in "lambda")
  views       view-set re-selection under the normalized MV-EAC (paper Section 5.4): No-Category,
              No-Entity, No-Topic (other two views at 1/2) and +Sentiment (4 views at 1/4), at
              MV-EAC's selected (lambda, beta)
  d1          entity controls: No-Entity with category and topic kept at 1/3, at MV-EAC's
              (lambda, beta); and a category-topic model (1/2 each) over its own lambda x beta grid
  simplex     the 28 weight vectors of the simplex with step 1/6, at MV-EAC's (lambda, beta)
  components  exploration components (hist-only, list-only) of MV-EAC, Topic-EAC and Entity-EAC at
              the lambda selected for the full method, over the beta grid (Section 5.3.5)

Output: $MVEAC_DATA_ROOT/results/rev2/val_grid_{model}_{config}.csv (resumable).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac import revision2 as R
from mveac.evaluation.metrics_runner import compute_all

OUT = C.RESULTS_DIR / "rev2"
V3 = list(C.MV_VIEWS)


def _run(task):
    model, spec = task
    t0 = time.time()
    s = R.STATE
    ranked = R.rerank(model, spec)
    m = compute_all(ranked, s["behaviors"], s["article_maps"], s["vocabs"], s["category_vectors"], C.K)
    return {"model": model, "config": spec.name, "method": spec.method, "lambda": spec.lam, "beta": spec.beta,
            **{f"w_{k}": v for k, v in spec.weights}, "normalize": spec.normalize, "ucb_mode": spec.ucb_mode,
            "_elapsed_s": round(time.time() - t0, 1), **m}


def _path(model, name):
    return OUT / f"val_grid_{model}_{name}.csv"


def _key(spec):
    return (round(spec.lam, 3), round(spec.beta, 4), tuple(round(v, 4) for _, v in spec.weights))


def _done(model, name):
    p = _path(model, name)
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    wc = sorted(c for c in d.columns if c.startswith("w_"))
    return {(round(l, 3), round(b, 4), tuple(round(x, 4) for x in ws))
            for l, b, *ws in d[["lambda", "beta"] + wc].itertuples(index=False)}


def _selected(model, config):
    b = pd.read_csv(OUT / "best_params_rev2.csv")
    r = b[(b.model == model) & (b.config == config)].iloc[0]
    return float(r["lambda"]), float(r.beta)


def build_tasks(tasks_wanted, models):
    tasks, done = [], {}

    def add(model, spec):
        if _key(spec) not in done.setdefault((model, spec.name), _done(model, spec.name)):
            tasks.append((model, spec))

    lb = list(itertools.product(C.LAMBDA_GRID, C.BETA_GRID_REV2))
    for model in models:
        if "eac" in tasks_wanted:
            for meth in C.GRID_METHODS:            # includes "mv_eac" = MV-EAC_nom
                for l, b in lb:
                    add(model, R.eac_spec(meth, meth, l, b))
            for l, b in itertools.product(C.LAMBDA_GRID, C.ENTITY_BETA_EXTENSION):
                add(model, R.eac_spec("entity_eac", "entity_eac", l, b))
        if "norm" in tasks_wanted:
            for l, b in lb:
                add(model, R.eac_spec("mv_eac_norm", "mv_eac_norm", l, b, normalize=True))
        if "baselines" in tasks_wanted:
            for meth, grid in C.BASELINE_GRIDS.items():
                for p in grid:
                    add(model, R.Spec(meth, "baseline", meth, float(p), 0.0))
        if {"views", "d1", "simplex", "components"} & set(tasks_wanted):
            ln, bn = _selected(model, "mv_eac_norm")
        if "views" in tasks_wanted:
            for drop in V3:
                add(model, R.eac_spec(f"norm_no_{drop}", "mv_eac_norm", ln, bn, {v: 0.5 for v in V3 if v != drop}, True))
            add(model, R.eac_spec("norm_plus_sentiment", "mv_eac_norm", ln, bn, C.SENTIMENT_ADDBACK_WEIGHTS, True))
        if "d1" in tasks_wanted:
            add(model, R.eac_spec("norm_no_entity_fixed", "mv_eac_norm", ln, bn, C.NO_ENTITY_FIXED_WEIGHTS, True))
            for l, b in lb:
                add(model, R.eac_spec("norm_cat_topic_grid", "mv_eac_norm", l, b, C.CAT_TOPIC_WEIGHTS, True))
        if "simplex" in tasks_wanted:
            for w in R.simplex_weights():
                add(model, R.eac_spec("norm_simplex", "mv_eac_norm", ln, bn, w, True))
        if "components" in tasks_wanted:
            for meth in C.COMPONENT_METHODS:
                lam, _ = _selected(model, meth)
                betas = C.BETA_GRID_REV2 + (C.ENTITY_BETA_EXTENSION[:-1] if meth == "entity_eac" else [])
                for mode in C.UCB_COMPONENTS:
                    for b in betas:
                        if b > 0:
                            add(model, R.eac_spec(f"c1_{meth}_{mode}", meth, lam, b, None,
                                                  meth == "mv_eac_norm", mode))
    # multi-view and entity tasks first, for load balance
    tasks.sort(key=lambda t: 0 if t[1].method in ("mv_eac", "mv_eac_norm", "entity_eac") else 1)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", required=True,
                    choices=["eac", "norm", "baselines", "views", "d1", "simplex", "components"])
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N validation impressions")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(a.task, a.models)
    print(f"pending tasks: {len(tasks)}", flush=True)
    if a.dry_run or not tasks:
        return
    R.load_state("val", a.models, need_embeddings="baselines" in a.task, limit=a.limit)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(_run, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f], repr(e), flush=True)
                continue
            p = _path(row["model"], row["config"])
            rows = pd.read_csv(p).to_dict("records") if p.exists() else []
            pd.DataFrame(rows + [row]).sort_values(["lambda", "beta"]).to_csv(p, index=False)
            n += 1
            print(f"[{n}/{len(tasks)}] {row['model']}/{row['config']} l={row['lambda']} b={row['beta']} "
                  f"NDCG={row['NDCG@K']:.4f} Ent={row['Entropy']:.4f} ERR={row['ERR@K']:.4f} "
                  f"TCI={row['TCI@K']:.4f} ETA {(time.time() - t0) / n * (len(tasks) - n) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
