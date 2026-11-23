#!/usr/bin/env python3
"""
Second revision, step 22 -- list-derived metrics of the revised paper (Section 4.5.4), computed from
the persisted test lists of step 21 (no reranking):

  @10 (sanity: equals step 21)          NDCG10 ILD10 ENT10 ERR10 TCI10
  @5                                     NDCG5 ILD5 ENT5 ERR5 TCI5
  position-weighted, w_r = 1/log2(r+1)  ILDpw ENTpw ERRpw TCIpw
  per-view calibration error C_KL       CKL_cat CKL_top CKL_ent CKL_sen
  regime flags                           nrms_nan (user without history: NRMS cannot score), n_cand

Output: results/rev2/derived/per_impression/<model>_<config>.parquet, results/rev2/derived/summary.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.metrics.accuracy import ndcg_at_k
from mveac.metrics.position import calibration_error, category_ild_entropy, position_weights, weighted_err, weighted_tci

REV2 = C.RESULTS_DIR / "rev2"
OUT = REV2 / "derived"
VIEWS = {"cat": "category", "top": "topic", "ent": "entity", "sen": "sentiment"}
G: dict = {}


def _load():
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        maps = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        prof = pickle.load(f)
    test = pd.read_parquet(C.PROC_DIR / "test_sub.parquet", columns=["impression_id", "user_id", "article_ids_clicked"])
    test = test.drop_duplicates("impression_id")
    sc = pd.read_parquet(C.score_path("nrms", "test"), columns=["impression_id", "score"])
    g = sc.groupby("impression_id")["score"]
    nan_all = g.apply(lambda x: x.isna().all())
    # only labels in the view's vocabulary count (as in the reranker)
    vsets = {k: set(vocabs[v]) for k, v in VIEWS.items()}
    vmaps = {k: {int(a): [x for x in xs if x in vsets[k]] for a, xs in maps[v].items()} for k, v in VIEWS.items()}
    # ERR counts every entity annotation of the article map, as the reranker-side metric of step 21 does
    G.update(vocabs=vocabs, maps=vmaps, ent_raw={int(a): xs for a, xs in maps["entity"].items() if xs}, prof={k: prof[v] for k, v in VIEWS.items()},
             clicks={int(i): set(int(a) for a in c) for i, c in zip(test.impression_id, test.article_ids_clicked)},
             i2u=dict(zip(test.impression_id.astype(int), test.user_id.astype(int))),
             nan_all=set(nan_all[nan_all].index.astype(int)), ncand=g.size().to_dict())


def _ckl(view, uid, arts):
    vocab = G["vocabs"][VIEWS[view]]
    p = G["prof"][view].get(uid)
    if p is None:
        p = np.full(len(vocab), 1.0 / len(vocab))
    return calibration_error(arts, p, G["maps"][view], vocab, alpha=C.CKL_ALPHA)


def _one(name):
    model = next(m for m in C.MODELS if name.startswith(m + "_"))
    z = np.load(REV2 / "lists" / f"{name}.npz")
    W = position_weights(C.K)
    cat, top, ent = G["maps"]["cat"], G["maps"]["top"], G["ent_raw"]
    ntop = len(G["vocabs"]["topic"])
    rows = []
    for iid, arr in zip(z["impression_id"], z["articles"]):
        iid = int(iid)
        arts = [int(a) for a in arr if a >= 0]
        uid = G["i2u"][iid]
        clicked = G["clicks"][iid]
        a5 = arts[:5]
        ild10, ent10 = category_ild_entropy(arts, cat)
        ild5, ent5 = category_ild_entropy(a5, cat)
        ildpw, entpw = category_ild_entropy(arts, cat, W[:len(arts)])
        rows.append((iid, ndcg_at_k(arts, clicked, 10), ild10, ent10, weighted_err(arts, ent), weighted_tci(arts, top, ntop),
                     ndcg_at_k(arts, clicked, 5), ild5, ent5, weighted_err(a5, ent), weighted_tci(a5, top, ntop),
                     ildpw, entpw, weighted_err(arts, ent, W), weighted_tci(arts, top, ntop, W),
                     _ckl("cat", uid, arts), _ckl("top", uid, arts), _ckl("ent", uid, arts), _ckl("sen", uid, arts),
                     iid in G["nan_all"], G["ncand"].get(iid, -1)))
    cols = ["impression_id", "NDCG10", "ILD10", "ENT10", "ERR10", "TCI10", "NDCG5", "ILD5", "ENT5", "ERR5", "TCI5",
            "ILDpw", "ENTpw", "ERRpw", "TCIpw", "CKL_cat", "CKL_top", "CKL_ent", "CKL_sen", "nrms_nan", "n_cand"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(OUT / "per_impression" / f"{name}.parquet", index=False)
    return {"name": name, "n": len(df), **df.drop(columns=["impression_id", "nrms_nan", "n_cand"]).mean().to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="subset of <model>_<config> names")
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    a = ap.parse_args()
    (OUT / "per_impression").mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (REV2 / "lists").glob("*.npz") if any(p.stem.startswith(m + "_") for m in C.MODELS)
                   and not p.stem.startswith("nrms_seed") and not p.stem.startswith("nrms_large"))
    if a.only:
        names = [n for n in names if n in set(a.only)]
    _load()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(_one, names))
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
