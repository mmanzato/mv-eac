#!/usr/bin/env python3
"""
Step 8 of 10 -- final test-set evaluation (500,000 impressions) for every
configuration reported in the paper.

For each base model, evaluates:
    original                                     uncalibrated base ranking
    {view}_eac / traditional_{view}_eac           for view in {category, entity, topic, sentiment}
    mv_eac / traditional_mv_eac                    the 3-view V* = {category, entity, topic}
    ablation_{no_category,no_entity,no_topic}      RQ2: drop one V* view, redistribute its weight
    mv4_at_mv3params                               RQ2: sentiment add-back (4 views @ 1/4, MV-EAC's lambda/beta)
    sweep_wtopic_{w}                               RQ4: continuous topic-weight sweep
                                                   (w_topic = 0.00 is the same configuration as ablation_no_topic)

With ``--nrms-tag <tag>`` (tag != seed0) only the NRMS reproducibility subset
{original, topic_eac, entity_eac, mv_eac, traditional_mv_eac, the three ablations, mv4_at_mv3params} is evaluated, at the
seed0 (lambda, beta), and every output file name carries the tag
(``per_impression_nrms_<tag>_<method>.csv``) so seed0 results are never overwritten.

at the (lambda, beta) selected by step 6 (``best_params.csv``). For every
configuration, writes one row per impression (with every metric --
everything downstream, including all statistical tests, is computed from
these files) plus the reranked top-K article-id list itself, and appends a
mean-over-impressions summary row.

Usage:
    python scripts/08_evaluate_test.py
    python scripts/08_evaluate_test.py --models most_popular   # one model only
    python scripts/08_evaluate_test.py --nrms-tag seed1        # NRMS reproducibility variant

Output (under $MVEAC_DATA_ROOT/results/ and .../lists/):
    per_impression_{model}_{method}.csv
    lists/{model}_{method}.npz            (impression_id + top-K article_id, for reuse)
    test_summary.csv                      one row per (model, method): mean of every metric
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
from mveac.calibration.eac import rerank_all
from mveac.evaluation.metrics_runner import per_impression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LISTS_DIR = C.RESULTS_DIR / "lists"
LISTS_DIR.mkdir(parents=True, exist_ok=True)
_SHARED: dict = {}


def _load_shared(models: list[str], nrms_tag: str) -> None:
    t0 = time.time()
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        article_maps = pickle.load(f)
    with open(C.SEM_DIR / "category_vectors.pkl", "rb") as f:
        category_vectors = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        all_profiles = pickle.load(f)
    user_profiles = {v: all_profiles[v] for v in C.CANDIDATE_VIEWS}
    del all_profiles

    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    scores, cand_counts = {}, {}
    for model in models:
        fname = f"scores_nrms_{nrms_tag}_test.parquet" if model == "nrms" else f"scores_{model}_test.parquet"
        df = pd.read_parquet(C.SCORES_DIR / fname)
        cand_counts[model] = df.groupby("impression_id").size().to_dict()
        scores[model] = {
            int(iid): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
            for iid, g in df.groupby("impression_id")
        }

    best = pd.read_csv(C.RESULTS_DIR / "best_params.csv")
    _SHARED.update(vocabs=vocabs, article_maps=article_maps, category_vectors=category_vectors,
                    user_profiles=user_profiles, test_sub=test_sub, scores=scores,
                    cand_counts=cand_counts, best=best)
    log.info("Shared state loaded in %.1fs (test impressions=%d)", time.time() - t0, len(test_sub))


def _base_model(model: str) -> str:
    """'nrms_seed1' -> 'nrms' (variants share seed0's hyperparameters and candidate sets)."""
    return model.split("_seed")[0].split("_large")[0]


def _best_param(model: str, method: str, column: str) -> float:
    best = _SHARED["best"]
    row = best[(best.model == _base_model(model)) & (best.method == method)]
    return float(row[column].iloc[0])


NRMS_VARIANT_METHODS = {"original", "topic_eac", "entity_eac", "mv_eac", "traditional_mv_eac",
                        "ablation_no_category", "ablation_no_entity", "ablation_no_topic", "mv4_at_mv3params"}


def _build_tasks(models: list[str], nrms_tag: str = "seed0") -> list[tuple]:
    """Return (model, output_name, rerank_method, lambda, beta, weights) tuples for
    every configuration reported in the paper (or the NRMS-variant subset)."""
    tasks = []
    for model in models:
        tasks.append((model, "original", "original", 0.0, 0.0, None))

        for view in ["category", "entity", "topic", "sentiment"]:
            method = f"{view}_eac"
            lam, beta = _best_param(model, method, "eac_lambda"), _best_param(model, method, "eac_beta")
            trad_lam = _best_param(model, method, "trad_lambda")
            tasks.append((model, method, method, lam, beta, None))
            tasks.append((model, f"traditional_{view}_eac", method, trad_lam, 0.0, None))

        mv_lam, mv_beta = _best_param(model, "mv_eac", "eac_lambda"), _best_param(model, "mv_eac", "eac_beta")
        mv_trad = _best_param(model, "mv_eac", "trad_lambda")
        tasks.append((model, "mv_eac", "mv_eac", mv_lam, mv_beta, C.MV_UNIFORM_WEIGHTS))
        tasks.append((model, "traditional_mv_eac", "mv_eac", mv_trad, 0.0, C.MV_UNIFORM_WEIGHTS))

        for name, weights in C.ABLATION_CONFIGS.items():
            tasks.append((model, f"ablation_{name}", "mv_eac", mv_lam, mv_beta, weights))
        tasks.append((model, "mv4_at_mv3params", "mv_eac", mv_lam, mv_beta, C.SENTIMENT_ADDBACK_WEIGHTS))

        for w_topic in C.WEIGHT_SWEEP:
            tasks.append((model, f"sweep_wtopic_{w_topic:.2f}", "mv_eac", mv_lam, mv_beta, C.sweep_weights(w_topic)))
    if nrms_tag != "seed0":
        tasks = [(f"nrms_{nrms_tag}", *t[1:]) for t in tasks
                 if t[0] == "nrms" and t[1] in NRMS_VARIANT_METHODS]
    return tasks


def _run_one(task: tuple) -> dict:
    model, output_name, rerank_method, lam, beta, weights = task
    per_path = C.RESULTS_DIR / f"per_impression_{model}_{output_name}.csv"
    if per_path.exists():
        return {"model": model, "method": output_name, "skipped": True}

    t0 = time.time()
    ranked = rerank_all(
        _SHARED["scores"][_base_model(model)], _SHARED["test_sub"], _SHARED["user_profiles"],
        _SHARED["article_maps"], _SHARED["vocabs"], method=rerank_method,
        lam=lam, beta=beta, k=C.K, weights=weights, epsilon=C.KL_EPSILON,
    )

    # Persist the reranked article-id lists (not just aggregate metrics) so any
    # future redefinition of a metric can be recomputed WITHOUT rerunning the
    # (expensive) reranking step.
    imp_ids = np.fromiter(ranked.keys(), dtype=np.int64, count=len(ranked))
    article_ids = np.full((len(ranked), C.K), -1, dtype=np.int64)
    for i, imp_id in enumerate(imp_ids):
        top_k = [aid for aid, _ in ranked[imp_id][:C.K]]
        article_ids[i, :len(top_k)] = top_k
    np.savez_compressed(LISTS_DIR / f"{model}_{output_name}.npz", impression_id=imp_ids, articles=article_ids)

    per = per_impression(ranked, _SHARED["test_sub"], _SHARED["article_maps"], _SHARED["vocabs"],
                         _SHARED["category_vectors"], C.K)
    candidate_counts = _SHARED["cand_counts"][_base_model(model)]
    rows = []
    for imp_id, metrics in per.items():
        row = dict(metrics)
        row["impression_id"] = imp_id
        row["n_candidates"] = int(candidate_counts.get(imp_id, 0))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(per_path, index=False)

    summary = {"model": model, "method": output_name, "lambda": lam, "beta": beta,
               "elapsed_s": round(time.time() - t0, 1)}
    for col in ("NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K", "HHI@K"):
        if col in df.columns:
            summary[col] = float(df[col].mean())
            summary[col + "__ru_gt_K"] = float(df.loc[df.n_candidates > C.K, col].mean())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    ap.add_argument("--nrms-tag", default="seed0",
                    help="Score which NRMS training run (see scripts/04_train_nrms.py --tag).")
    args = ap.parse_args()

    if not (C.RESULTS_DIR / "best_params.csv").exists():
        log.error("best_params.csv missing -- run scripts/06_select_hyperparameters.py first.")
        return

    _load_shared(args.models, args.nrms_tag)
    tasks = [t for t in _build_tasks(args.models, args.nrms_tag) if _base_model(t[0]) in args.models]
    log.info("Configurations: %d (workers=%d)", len(tasks), args.workers)

    summary_path = C.RESULTS_DIR / (
        "test_summary.csv" if args.nrms_tag == "seed0" else f"test_summary_nrms_{args.nrms_tag}.csv"
    )
    done = set()
    if summary_path.exists():
        d = pd.read_csv(summary_path)
        done = set(zip(d.model, d.method))
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks if (t[0], t[1]) not in done}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
            except Exception:
                log.exception("Configuration %s/%s failed", task[0], task[1])
                continue
            if result.get("skipped"):
                log.info("[%d/%d] %s/%s already done", i, len(futures), task[0], task[1])
                continue
            rows = [r for r in rows if (r["model"], r["method"]) != (result["model"], result["method"])]
            rows.append(result)
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            log.info("[%d/%d] %s/%s NDCG=%.4f ERR=%.4f TCI=%.4f",
                      i, len(futures), result["model"], result["method"],
                      result.get("NDCG@K", float("nan")), result.get("ERR@K", float("nan")),
                      result.get("TCI@K", float("nan")))

    log.info("Test-set evaluation complete -> %s", summary_path)


if __name__ == "__main__":
    main()
