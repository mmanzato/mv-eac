#!/usr/bin/env python3
"""
Second revision, step 21 -- test-set evaluation (500,000 impressions) of every configuration of the
revised paper, at the step-20 selections. MCF is not re-run (its selections did not change); step 23
reads its per-impression files from step 07.

Per base model (most_popular, itemknn, nrms):
  original
  <v>_eac, tradcal_<v>                 v in {category, entity, topic, sentiment}
  mv_eac_norm, tradcal_mv_norm         the paper's MV-EAC and Trad-Cal-MV (normalized)
  mv_eac, tradcal_mv                   MV-EAC_nom and Trad-Cal-MV_nom
  ablation_no_{category,entity,topic}, plus_sentiment     at MV-EAC's (lambda, beta)
  d1_no_entity_fixed, d1_cat_topic_tuned                  entity controls
  sweep_wtopic_{0.15,0.50,0.70,0.85,1.00}                 at MV-EAC's (lambda, beta)
  c1_<method>_{hist,list}              exploration components (MV-EAC, Topic-EAC, Entity-EAC)
  mmr, xquad, dpp, dedup               diversification baselines
  eps<e>_mv_eac_norm, eps<e>_tradcal_mv_norm   KL smoothing e in {1e-4, 1e-6, 1e-8}
Per additional NRMS run (``--models nrms_seed1 nrms_seed2 nrms_large1M``), at the primary NRMS
selections: original, mv_eac_norm, tradcal_mv_norm, topic_eac, entity_eac, ablation_no_*, plus_sentiment.

Output (resumable): results/rev2/lists/<model>_<config>.npz, results/rev2/per_impression/<model>_<config>.parquet,
results/rev2/test_summary_rev2.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac import revision2 as R
from mveac.evaluation.metrics_runner import per_impression

OUT = C.RESULTS_DIR / "rev2"
V3 = list(C.MV_VIEWS)


def build_tasks(models):
    b1 = pd.read_csv(OUT / "best_params_rev2.csv")
    b2 = pd.read_csv(OUT / "best_params_rev2b.csv")

    def sel(model, config, table=b1):
        r = table[(table.model == model) & (table.config == config)].iloc[0]
        return float(r["lambda"]), float(r.beta)

    tasks = []
    for model in models:
        base = "nrms" if model.startswith("nrms_") else model
        primary = model in C.MODELS
        add = lambda spec: tasks.append((model, spec))  # noqa: E731
        nl = sel(base, "mv_eac_norm")
        add(R.Spec("original", "original", "original"))
        add(R.eac_spec("mv_eac_norm", "mv_eac_norm", *nl, normalize=True))
        add(R.eac_spec("tradcal_mv_norm", "mv_eac_norm", *sel(base, "tradcal_mv_norm"), normalize=True))
        for v in (["category", "entity", "topic", "sentiment"] if primary else ["topic", "entity"]):
            add(R.eac_spec(f"{v}_eac", f"{v}_eac", *sel(base, f"{v}_eac")))
            if primary:
                add(R.eac_spec(f"tradcal_{v}", f"{v}_eac", *sel(base, f"tradcal_{v}")))
        for x in V3:
            add(R.eac_spec(f"ablation_no_{x}", "mv_eac_norm", *nl, {v: 0.5 for v in V3 if v != x}, True))
        add(R.eac_spec("plus_sentiment", "mv_eac_norm", *nl, C.SENTIMENT_ADDBACK_WEIGHTS, True))
        if not primary:
            continue
        add(R.eac_spec("mv_eac", "mv_eac", *sel(model, "mv_eac")))
        add(R.eac_spec("tradcal_mv", "mv_eac", *sel(model, "tradcal_mv")))
        add(R.eac_spec("d1_no_entity_fixed", "mv_eac_norm", *nl, C.NO_ENTITY_FIXED_WEIGHTS, True))
        add(R.eac_spec("d1_cat_topic_tuned", "mv_eac_norm", *sel(model, "cat_topic_tuned", b2), C.CAT_TOPIC_WEIGHTS, True))
        for wt in [w for w in C.WEIGHT_SWEEP if w > 0]:
            add(R.eac_spec(f"sweep_wtopic_{wt:.2f}", "mv_eac_norm", *nl, C.sweep_weights(wt), True))
        for meth in C.COMPONENT_METHODS:
            for mode in C.UCB_COMPONENTS:
                add(R.eac_spec(f"c1_{meth}_{mode}", meth, *sel(model, f"c1_{meth}_{mode}", b2), None,
                               meth == "mv_eac_norm", mode))
        for bl in C.BASELINE_GRIDS:
            add(R.Spec(bl, "baseline", bl, sel(model, bl)[0], 0.0))
        for e in C.EPS_GRID_REV2:
            add(R.eac_spec(f"eps{e:.0e}_mv_eac_norm", "mv_eac_norm", *nl, normalize=True, epsilon=e))
            add(R.eac_spec(f"eps{e:.0e}_tradcal_mv_norm", "mv_eac_norm", *sel(model, "tradcal_mv_norm"),
                           normalize=True, epsilon=e))
    return tasks


def _run(task):
    model, spec = task
    t0 = time.time()
    s = R.STATE
    ranked = R.rerank(model, spec)
    ids, arts = R.lists_to_arrays(ranked)
    np.savez_compressed(OUT / "lists" / f"{model}_{spec.name}.npz", impression_id=ids, articles=arts)
    per = per_impression(ranked, s["behaviors"], s["article_maps"], s["vocabs"], s["category_vectors"], C.K)
    df = pd.DataFrame.from_dict(per, orient="index")
    df.index.name = "impression_id"
    df = df.reset_index().drop(columns=[c for c in ("AUC",) if c in df.columns])  # not a valid AUC on cut lists
    df["n_cand"] = df.impression_id.map(s["counts"][model]).fillna(0).astype(int)
    df.to_parquet(OUT / "per_impression" / f"{model}_{spec.name}.parquet", index=False)
    row = {"model": model, "config": spec.name, "method": spec.method, "lambda": spec.lam, "beta": spec.beta,
           "weights": json.dumps(spec.weight_dict()) if spec.kind == "eac" else "", "normalize": spec.normalize,
           "ucb_mode": spec.ucb_mode, "eps": spec.epsilon, "elapsed_s": round(time.time() - t0, 1)}
    for c in ["NDCG@K", "MRR", "ILD@K", "Entropy", "ERR@K", "TCI@K"]:
        if c in df.columns:
            row[c] = float(df[c].mean())
            row[c + "__ru_gt_K"] = float(df.loc[df.n_cand > C.K, c].mean())
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    ap.add_argument("--summary", default="test_summary_rev2.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for d in ("lists", "per_impression"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    tasks = [t for t in build_tasks(a.models) if not (OUT / "per_impression" / f"{t[0]}_{t[1].name}.parquet").exists()]
    tasks.sort(key=lambda t: 0 if t[1].method in ("mv_eac", "mv_eac_norm", "entity_eac") else 1)
    print(f"pending configurations: {len(tasks)}", flush=True)
    if a.dry_run:
        for m, s in tasks:
            print(m, s)
        return
    if not tasks:
        return
    R.load_state("test", a.models, need_embeddings=any(s.kind == "baseline" for _, s in tasks))
    summ = OUT / a.summary
    rows = pd.read_csv(summ).to_dict("records") if summ.exists() else []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(_run, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f][0], futs[f][1].name, repr(e), flush=True)
                continue
            rows = [x for x in rows if (x["model"], x["config"]) != (r["model"], r["config"])] + [r]
            pd.DataFrame(rows).to_csv(summ, index=False)
            print(f"[{i}/{len(futs)}] {r['model']}/{r['config']} NDCG={r['NDCG@K']:.4f} Ent={r['Entropy']:.4f} "
                  f"ERR={r['ERR@K']:.4f} TCI={r['TCI@K']:.4f} [{r['elapsed_s']}s] {(time.time() - t0) / 60:.0f} min",
                  flush=True)


if __name__ == "__main__":
    main()
