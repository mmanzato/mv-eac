#!/usr/bin/env python3
"""
Second revision, step 24 -- the two decisions taken on the validation set (50,000 impressions),
paired per impression at the step-20 selections.

  --part norm    adoption rule for the per-view normalization (paper Section 5.8, Supplementary
                 Table S7), fixed before the comparison: the normalized MV-EAC replaces MV-EAC_nom
                 only if it is not worse on any MAUT metric by a Holm-significant |d| >= 0.10 and
                 better on at least one, in every base model. Each formulation at its own selection.
  --part views   view-set re-selection under the normalized MV-EAC with the paper's inclusion
                 criterion (Section 5.4; 3 ablations + sentiment add-back x 5 metrics x 3 models =
                 60 tests, Holm on their own), and the two entity controls (separate Holm family):
                 a view is excluded iff removing it is never worse by a Holm-significant |d| >= 0.10
                 on any metric in any base model; sentiment is added iff adding it improves some
                 metric by a Holm-significant |d| >= 0.10 in some base model.

Output: results/rev2/val_compare_norm.csv, results/rev2/val_views_norm.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac import revision2 as R
from mveac.evaluation.metrics_runner import per_impression

OUT = C.RESULTS_DIR / "rev2"
V3 = list(C.MV_VIEWS)
MET_NORM = {"NDCG@K": +1, "Entropy": +1, "ERR@K": -1, "TCI@K": -1}            # the MAUT metrics
MET_VIEWS = {"NDCG@K": +1, "ILD@K": +1, "Entropy": +1, "ERR@K": -1, "TCI@K": -1}


def _run(task):
    model, spec, metrics = task
    s = R.STATE
    p = per_impression(R.rerank(model, spec), s["behaviors"], s["article_maps"], s["vocabs"], s["category_vectors"], C.K)
    df = pd.DataFrame.from_dict(p, orient="index").rename(columns={"Entropy@K": "Entropy"})[list(metrics)]
    return (model, spec.name), df


def holm(p):
    o = np.argsort(p)
    m = len(p)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def paired(a, b, metric, sign):
    """d_z of (b - a) and the Wilcoxon p-value on the non-zero differences."""
    a, b = a.align(b, join="inner", axis=0)
    d = (b[metric] - a[metric]).astype(float).to_numpy()
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else 0.0
    p = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 0 else 1.0
    return len(d), a[metric].mean(), b[metric].mean(), dz, sign * dz, p


def verdicts(df, better, worse):
    df["p_holm"] = holm(df.p.values)
    df["verdict"] = np.where((df.p_holm < .05) & (df.d_better >= .10), better,
                             np.where((df.p_holm < .05) & (df.d_better <= -.10), worse, "negligible/ns"))
    return df


def part_norm(workers):
    sel = pd.read_csv(OUT / "best_params_rev2.csv")
    get = lambda m, c: tuple(sel[(sel.model == m) & (sel.config == c)][["lambda", "beta"]].iloc[0].astype(float))  # noqa: E731
    tasks = []
    for m in C.MODELS:
        tasks.append((m, R.eac_spec("mv_eac", "mv_eac", *get(m, "mv_eac")), MET_NORM))
        tasks.append((m, R.eac_spec("mv_eac_norm", "mv_eac_norm", *get(m, "mv_eac_norm"), normalize=True), MET_NORM))
    with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=mp.get_context("fork")) as ex:
        res = dict(ex.map(_run, tasks))
    rows = []
    for m in C.MODELS:
        for metric, sgn in MET_NORM.items():
            n, ma, mb, dz, db, p = paired(res[(m, "mv_eac")], res[(m, "mv_eac_norm")], metric, sgn)
            rows.append({"model": m, "metric": metric, "n": n, "mean_nom": ma, "mean_norm": mb, "d_z": dz,
                         "d_better": db, "p": p})
    df = verdicts(pd.DataFrame(rows), "Norm better", "Norm WORSE")
    df.to_csv(OUT / "val_compare_norm.csv", index=False)
    print(df.round(4).to_string(index=False))
    for m in C.MODELS:
        v = df[df.model == m].verdict
        print(m, "ADOPT normalization" if not (v == "Norm WORSE").any() and (v == "Norm better").any() else "keep MV-EAC_nom")


def part_views(workers):
    sel = pd.read_csv(OUT / "best_params_rev2.csv")
    selb = pd.read_csv(OUT / "best_params_rev2b.csv")
    tasks = []
    for m in C.MODELS:
        l, b = sel[(sel.model == m) & (sel.config == "mv_eac_norm")][["lambda", "beta"]].iloc[0].astype(float)
        lc, bc = selb[(selb.model == m) & (selb.config == "cat_topic_tuned")][["lambda", "beta"]].iloc[0].astype(float)
        specs = [R.eac_spec("full", "mv_eac_norm", l, b, normalize=True)]
        specs += [R.eac_spec(f"no_{x}", "mv_eac_norm", l, b, {v: 0.5 for v in V3 if v != x}, True) for x in V3]
        specs += [R.eac_spec("plus_sentiment", "mv_eac_norm", l, b, C.SENTIMENT_ADDBACK_WEIGHTS, True),
                  R.eac_spec("no_entity_fixed", "mv_eac_norm", l, b, C.NO_ENTITY_FIXED_WEIGHTS, True),
                  R.eac_spec("cat_topic_tuned", "mv_eac_norm", lc, bc, C.CAT_TOPIC_WEIGHTS, True)]
        tasks += [(m, s, MET_VIEWS) for s in specs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        res = dict(ex.map(_run, tasks))

    def family(name, variants):
        rows = []
        for m in C.MODELS:
            for v in variants:
                for metric, sgn in MET_VIEWS.items():
                    n, ma, mb, dz, db, p = paired(res[(m, "full")], res[(m, v)], metric, sgn)
                    rows.append({"family": name, "model": m, "variant": v, "metric": metric, "n": n,
                                 "mean_full": ma, "mean_variant": mb, "d_z": dz, "d_better": db, "p": p})
        return verdicts(pd.DataFrame(rows), "variant better", "variant WORSE")

    views = family("views", ["no_category", "no_entity", "no_topic", "plus_sentiment"])
    d1 = family("d1", ["no_entity_fixed", "cat_topic_tuned"])
    pd.concat([views, d1]).to_csv(OUT / "val_views_norm.csv", index=False)
    print("View criterion under the normalized MV-EAC:")
    for x in V3:
        worse = views[(views.variant == f"no_{x}") & (views.verdict == "variant WORSE")]
        print(f"  {x}: {'RETAIN' if len(worse) else 'EXCLUDE'} "
              + ", ".join(f"{r.model}/{r.metric} d={r.d_z:+.3f}" for r in worse.itertuples()))
    better = views[(views.variant == "plus_sentiment") & (views.verdict == "variant better")]
    print(f"  sentiment: {'ADD' if len(better) else 'EXCLUDE'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["norm", "views"], required=True)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    a = ap.parse_args()
    R.load_state("val", C.MODELS)
    part_norm(a.workers) if a.part == "norm" else part_views(a.workers)


if __name__ == "__main__":
    main()
