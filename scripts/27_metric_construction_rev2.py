#!/usr/bin/env python3
"""
Second revision, step 27 -- construct-validity checks of ERR@K / TCI@K (Supplementary Section S7, first
paragraph; paper Section 4.5) for the revised configurations (normalized MV-EAC / Trad-Cal-MV, d2 selections), from the persisted test lists.
Per-impression measures as in step 18. Paired Cohen's d_z (raw sign) + Wilcoxon/Holm.
Output: results/rev2/metric_construction/{per_impression_*.parquet, summary.csv, comparisons.csv}
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, pickle, sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
from mveac.evaluation.stats import cohens_d as dz, wilcoxon_p as wilc


def holm_adj(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


class MV:  # shared state of the worker processes
    _G: dict = {}

LISTS = C.RESULTS_DIR / "rev2" / "lists"
OUT = C.RESULTS_DIR / "rev2" / "metric_construction"
METHODS = ["original", "tradcal_entity", "entity_eac", "tradcal_topic", "topic_eac", "category_eac",
           "tradcal_mv_norm", "mv_eac_norm", "ablation_no_entity", "ablation_no_topic", "ablation_no_category"]
MEAS = ["ERR", "ERR_art", "ENT_ART", "MENTIONS", "TCI", "TCI_mass"]
COMPS = [("entity_eac", "tradcal_entity", ["ERR", "ERR_art", "ENT_ART", "MENTIONS"]),
         ("topic_eac", "tradcal_topic", ["TCI", "TCI_mass"]),
         ("mv_eac_norm", "tradcal_mv_norm", ["ERR", "ERR_art", "ENT_ART", "MENTIONS", "TCI", "TCI_mass"]),
         ("mv_eac_norm", "topic_eac", ["TCI", "TCI_mass"]),
         ("mv_eac_norm", "entity_eac", ["ERR", "ERR_art"]),
         ("ablation_no_entity", "mv_eac_norm", ["ERR", "ERR_art", "ENT_ART"]),
         ("ablation_no_topic", "mv_eac_norm", ["TCI", "TCI_mass"])]


def _one(task):
    model, method = task
    z = np.load(LISTS / f"{model}_{method}.npz")
    ent, top, N = MV._G["ent"], MV._G["top"], MV._G["N"]
    rows = []
    for iid, arts in zip(z["impression_id"], z["articles"]):
        arts = [int(a) for a in arts if a >= 0]
        ments = [e for a in arts for e in ent.get(a, ())]
        sets = [set(ent.get(a, ())) for a in arts]
        n_set = sum(len(s) for s in sets)
        uni = set().union(*sets) if sets else set()
        err = (len(ments) - len(uni)) / len(ments) if ments else 0.0
        err_art = (n_set - len(uni)) / n_set if n_set else 0.0
        cnt, mass = {}, {}
        for a in arts:
            labs = top.get(a, ())
            for x in labs:
                cnt[x] = cnt.get(x, 0) + 1
                mass[x] = mass.get(x, 0) + 1.0 / len(labs)
        def tci(d):
            s = sum(d.values())
            if s == 0:
                return 0.0
            h = sum((v / s) ** 2 for v in d.values())
            return min(1.0, max(0.0, (h - 1 / N) / (1 - 1 / N)))
        rows.append((int(iid), err, err_art, sum(1 for s in sets if s), len(ments), tci(cnt), tci(mass)))
    df = pd.DataFrame(rows, columns=["impression_id"] + MEAS)
    df.to_parquet(OUT / f"per_impression_{model}_{method}.parquet", index=False)
    return {"model": model, "method": method, **df.drop(columns="impression_id").mean().to_dict()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    MV._G.update(ent={int(k): v for k, v in maps["entity"].items() if v},
                 top={int(k): v for k, v in maps["topic"].items() if v}, N=len(vocabs["topic"]))
    tasks = [(m, meth) for m in C.MODELS for meth in METHODS]
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("fork")) as ex:
        summ = pd.DataFrame(list(ex.map(_one, tasks)))
    summ.to_csv(OUT / "summary.csv", index=False)
    rows = []
    for model in C.MODELS:
        P = {m: pd.read_parquet(OUT / f"per_impression_{model}_{m}.parquet").set_index("impression_id") for m in METHODS}
        for a, b, ms in COMPS:
            for meas in ms:
                x = (P[a][meas] - P[b][meas].reindex(P[a].index)).values
                rows.append({"model": model, "comparison": f"{a} vs {b}", "measure": meas,
                             "mean_a": P[a][meas].mean(), "mean_b": P[b][meas].mean(), "d": dz(x), "p": wilc(x)})
    cmp_ = pd.DataFrame(rows); cmp_["p_holm"] = holm_adj(cmp_["p"].values)
    cmp_.to_csv(OUT / "comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(cmp_.pivot_table(index=["comparison", "measure"], columns="model", values="d").round(3).to_string())
    print("max p_holm:", cmp_.p_holm.max())


if __name__ == "__main__":
    main()
