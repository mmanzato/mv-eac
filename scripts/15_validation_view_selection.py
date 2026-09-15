#!/usr/bin/env python3
"""
View selection on the validation set (paper Section 5.4, "View-inclusion criterion").

The view set V* is fixed on development data. For each base model, at MV-EAC's
(lambda, beta) from step 6 (best_params.csv), reranks the 50,000-impression validation
subsample with:
    mv_eac                                        V* = {category, entity, topic} @ 1/3
    ablation_{no_category,no_entity,no_topic}     drop one view, remaining two @ 1/2
    mv4_at_mv3params                              sentiment add-back, four views @ 1/4
and tests every variant against mv_eac on the five metrics (Wilcoxon signed-rank +
Cohen's d_z, impression- and user-level) as one Holm family of 60 tests. It then
applies the view-inclusion criterion: a candidate view is excluded if the model without
it is never worse by a non-negligible (|d| >= 0.10) Holm-significant effect on any metric
in any base model. The test-set runs of steps 8-9 (families E and H) confirm the decision.

Usage:
    python scripts/15_validation_view_selection.py [--workers 6] [--n-impressions N]

Output (under $MVEAC_DATA_ROOT/results/):
    val_selection/per_impression_{model}_{config}.csv
    val_view_selection_stats.csv         one row per (model, variant, metric)
"""
from __future__ import annotations

import os

# One BLAS thread per worker, set before numpy is imported: forked workers that inherit
# a multi-threaded Accelerate/OpenBLAS state can crash on macOS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

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
from mveac.evaluation.stats import compare, holm_bonferroni

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METRICS = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
MINIMIZE_METRICS = {"ERR@K", "TCI@K"}
CONFIGS = {
    "mv_eac": C.MV_UNIFORM_WEIGHTS,
    **{f"ablation_{name}": weights for name, weights in C.ABLATION_CONFIGS.items()},
    "mv4_at_mv3params": C.SENTIMENT_ADDBACK_WEIGHTS,
}
# variant -> the candidate view whose inclusion it tests
TESTED_VIEW = {"ablation_no_category": "category", "ablation_no_entity": "entity",
               "ablation_no_topic": "topic", "mv4_at_mv3params": "sentiment"}
_SHARED: dict = {}


def _out_dir(smoke: bool) -> Path:
    path = C.RESULTS_DIR / ("val_selection_smoke" if smoke else "val_selection")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_shared(models: list[str], n_impressions: int | None) -> None:
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

    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    if n_impressions:
        val_sub = val_sub.head(n_impressions)
    scores = {}
    for model in models:
        df = pd.read_parquet(C.SCORES_DIR / f"scores_{model}_val.parquet")
        df = df[df["impression_id"].isin(val_sub["impression_id"])]
        scores[model] = {
            int(iid): [(int(r.article_id), float(r.score)) for r in g.sort_values("rank").itertuples(index=False)]
            for iid, g in df.groupby("impression_id")
        }
    best = pd.read_csv(C.RESULTS_DIR / "best_params.csv")
    params = {m: (float(best[(best.model == m) & (best.method == "mv_eac")].eac_lambda.iloc[0]),
                  float(best[(best.model == m) & (best.method == "mv_eac")].eac_beta.iloc[0])) for m in models}
    _SHARED.update(vocabs=vocabs, article_maps=article_maps, category_vectors=category_vectors,
                   user_profiles=user_profiles, val_sub=val_sub, scores=scores, params=params,
                   out=_out_dir(bool(n_impressions)))
    log.info("Shared state loaded in %.1fs (validation impressions=%d)", time.time() - t0, len(val_sub))


def _run_one(task: tuple[str, str]) -> str:
    model, config = task
    path = _SHARED["out"] / f"per_impression_{model}_{config}.csv"
    if path.exists():
        return f"{model}/{config} (cached)"
    lam, beta = _SHARED["params"][model]
    ranked = rerank_all(_SHARED["scores"][model], _SHARED["val_sub"], _SHARED["user_profiles"],
                        _SHARED["article_maps"], _SHARED["vocabs"], method="mv_eac", lam=lam, beta=beta,
                        k=C.K, weights=CONFIGS[config], epsilon=C.KL_EPSILON)
    per = per_impression(ranked, _SHARED["val_sub"], _SHARED["article_maps"], _SHARED["vocabs"],
                         _SHARED["category_vectors"], C.K)
    pd.DataFrame([{"impression_id": imp_id, **metrics} for imp_id, metrics in per.items()]).to_csv(path, index=False)
    return f"{model}/{config}"


def _statistics(models: list[str]) -> pd.DataFrame:
    out = _SHARED["out"]
    users = _SHARED["val_sub"].drop_duplicates("impression_id").set_index("impression_id")["user_id"]
    rows = []
    for model in models:
        base = pd.read_csv(out / f"per_impression_{model}_mv_eac.csv").set_index("impression_id")
        for variant, view in TESTED_VIEW.items():
            other = pd.read_csv(out / f"per_impression_{model}_{variant}.csv").set_index("impression_id")
            common = base.index.intersection(other.index)
            for metric in METRICS:
                result = compare(other.loc[common, metric], base.loc[common, metric], users.reindex(common),
                                 minimize=metric in MINIMIZE_METRICS)
                rows.append({"model": model, "variant": variant, "tested_view": view, "metric": metric, **result})
    stats = pd.DataFrame(rows)
    stats["significant_holm"] = holm_bonferroni(stats["p_impression"].fillna(1).values, C.ALPHA)
    stats["significant_user_holm"] = holm_bonferroni(stats["p_user"].fillna(1).values, C.ALPHA)
    meaningful = stats["d_impression"].abs() >= C.EFFECT_SIZE_THRESHOLDS["negligible"]
    variant_better = stats["d_impression_improvement"] > 0
    # The model WITHOUT the tested view is worse than the one WITH it when: an ablation
    # (view removed) is worse than MV-EAC, or the add-back (view added) is better than MV-EAC.
    is_addback = stats["variant"] == "mv4_at_mv3params"
    stats["without_view_worse"] = stats["significant_holm"] & meaningful & np.where(is_addback, variant_better, ~variant_better)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    ap.add_argument("--n-impressions", type=int, default=None,
                    help="use only the first N validation impressions (smoke test)")
    args = ap.parse_args()

    _load_shared(args.models, args.n_impressions)
    tasks = [(m, c) for m in args.models for c in CONFIGS]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        for future in as_completed([pool.submit(_run_one, t) for t in tasks]):
            log.info("done %s", future.result())

    stats = _statistics(args.models)
    name = "val_view_selection_stats_smoke.csv" if args.n_impressions else "val_view_selection_stats.csv"
    stats.to_csv(C.RESULTS_DIR / name, index=False)

    log.info("Holm-significant: %d of %d tests", int(stats.significant_holm.sum()), len(stats))
    for view, sub in stats.groupby("tested_view", sort=False):
        hits = sub[sub.without_view_worse]
        decision = "RETAIN" if len(hits) else "EXCLUDE"
        detail = "; ".join(f"{r.model} {r.metric} d={r.d_impression:+.3f}" for r in hits.itertuples()) or "never worse"
        log.info("%-9s -> %s  (without it: %s)", view, decision, detail)


if __name__ == "__main__":
    main()
