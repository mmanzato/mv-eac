#!/usr/bin/env python3
"""
Second revision, step 31 -- MIND-small: validation grid -> MAUT selection -> test evaluation, with the EB-NeRD code paths.

  --stage grid    20,000 validation impressions; lambda in {0.1,...,0.9} x beta in the EB-NeRD revised grid
                  (0 ... 10; Entity-EAC also {20, 50, 100, inf}) for Category-/Topic-/Entity-EAC and MV-EAC-Norm
                  (3 views, uniform 1/3, per-step min-max normalization); xQuAD (the strongest B1 baseline on
                  EB-NeRD) over its EB-NeRD grid. Rerankers: mveac.calibration.eac.rerank_all_general and
                  mveac.calibration.baselines; metrics: mveac.evaluation.metrics_runner.
  --stage select  same MAUT (4 metrics, equal weights, 0.003 tolerance, smallest beta then lambda); Trad-Cal-X
                  = beta = 0 slice of the X grid; grid-top selections are flagged.
  --stage test    30,000 test impressions: Original, Trad-Cal-{category, topic, entity, MV-Norm},
                  {Category, Topic, Entity}-EAC, MV-EAC-Norm, xQuAD; per-impression metrics + persisted top-10 lists.
Output: $MVEAC_DATA_ROOT/mind/{grids, best_params_mind.csv, test/}
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, itertools, json, pickle, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
from mveac.calibration.baselines import rerank_all_baseline
from mveac.calibration.eac import rerank_all_general
from mveac.evaluation.maut import maut_score
from mveac.evaluation.metrics_runner import compute_all as compute_all_v2, per_impression as per_impression_v2


def maut(g, _weights=None):
    return maut_score(g)


ROOT = C.DATA_ROOT / "mind"
P, S, GR, TE = ROOT / "processed", ROOT / "scores", ROOT / "grids", ROOT / "test"
MODELS = ["most_popular", "nrms"]
V3 = ["category", "entity", "topic"]
LAMBDAS = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
BETA_ENTITY = BETA + [20.0, 50.0, 100.0, float("inf")]
XQUAD = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
EAC = ["category_eac", "topic_eac", "entity_eac", "mv_eac_norm"]
W = None; TOL = C.PARSIMONY_TOL
_G: dict = {}


def init(split):
    vocabs = json.load(open(P / "vocabs.json"))
    maps = pickle.load(open(P / "article_maps.pkl", "rb"))
    prof = pickle.load(open(P / f"user_profiles_{split}.pkl", "rb"))
    # metrics_v2 also reports an auxiliary sentiment entropy; MIND has no sentiment view -> empty map
    # (that column is meaningless on MIND and is not used; no reranker sees it: profiles hold 3 views only)
    maps.setdefault("sentiment", {}); vocabs.setdefault("sentiment", ["na"])
    ci = {c: i for i, c in enumerate(vocabs["category"])}
    catv = {}
    for a, cs in maps["category"].items():
        v = np.zeros(len(ci)); v[[ci[c] for c in cs]] = 1.0; catv[int(a)] = v
    sub = pd.read_parquet(P / f"{split}_sub.parquet")
    scores = {}
    for m in MODELS:
        df = pd.read_parquet(S / f"scores_{m}_{split}.parquet")
        scores[m] = {int(i): [(int(a), float(s)) for a, s in zip(g.article_id, g.score)]
                     for i, g in df.sort_values(["impression_id", "rank"]).groupby("impression_id")}
    _G.update(vocabs=vocabs, maps=maps, prof=prof, catv=catv, sub=sub, scores=scores)


def rerank(model, method, lam, beta):
    g = _G
    if method == "original":
        return {i: c[:C.K] for i, c in g["scores"][model].items()}
    if method == "xquad":
        return rerank_all_baseline("xquad", lam, g["scores"][model], g["sub"], g["prof"], g["maps"], g["vocabs"], C.K, {})
    if method == "mv_eac_norm":
        return rerank_all_general(g["scores"][model], g["sub"], g["prof"], g["maps"], g["vocabs"],
                                  {v: 1 / 3 for v in V3}, lam, beta, C.K, normalize=True)
    return rerank_all_general(g["scores"][model], g["sub"], g["prof"], g["maps"], g["vocabs"],
                              {method.replace("_eac", ""): 1.0}, lam, beta, C.K, normalize=False)


def _grid_task(t):
    model, method, lam, beta = t; t0 = time.time()
    r = rerank(model, method, lam, beta)
    m = compute_all_v2(r, _G["sub"], _G["maps"], _G["vocabs"], _G["catv"], C.K)
    return {"model": model, "method": method, "lambda": lam, "beta": beta, "_elapsed_s": round(time.time() - t0, 1), **m}


def _test_task(t):
    model, config, method, lam, beta = t; t0 = time.time()
    r = rerank(model, method, lam, beta)
    p = per_impression_v2(r, _G["sub"], _G["maps"], _G["vocabs"], _G["catv"], C.K)
    df = pd.DataFrame.from_dict(p, orient="index").rename_axis("impression_id").reset_index()
    df = df.rename(columns={"Entropy@K": "Entropy"}).drop(columns=["AUC"], errors="ignore")
    ncand = {i: len(c) for i, c in _G["scores"][model].items()}
    df["n_cand"] = df.impression_id.map(ncand)
    df.to_parquet(TE / "per_impression" / f"{model}_{config}.parquet", index=False)
    ids = np.array(sorted(r)); arts = np.full((len(ids), C.K), -1, np.int64)
    for j, i in enumerate(ids):
        a = [x for x, _ in r[i][:C.K]]; arts[j, :len(a)] = a
    np.savez_compressed(TE / "lists" / f"{model}_{config}.npz", impression_id=ids, articles=arts)
    cols = ["NDCG@K", "MRR", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
    return {"model": model, "config": config, "method": method, "lambda": lam, "beta": beta,
            "elapsed_s": round(time.time() - t0, 1), **df[cols].mean().to_dict(),
            **{f"{c}__ru_gt_K": df.loc[df.n_cand > C.K, c].mean() for c in cols}}


def pool(fn, tasks, workers, on_done):
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(fn, t): t for t in tasks}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                row = f.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f], repr(e), flush=True); continue
            on_done(row, n, len(tasks))


def stage_grid(workers):
    GR.mkdir(parents=True, exist_ok=True); init("val")
    path = lambda m, meth: GR / f"val_grid_mind_{m}_{meth}.csv"
    tasks = []
    for m in MODELS:
        for meth in EAC + ["xquad"]:
            p = path(m, meth)
            done = set(zip(pd.read_csv(p)["lambda"].round(3), pd.read_csv(p)["beta"].round(4))) if p.exists() else set()
            grid = ([(x, 0.0) for x in XQUAD] if meth == "xquad" else
                    itertools.product(LAMBDAS, BETA_ENTITY if meth == "entity_eac" else BETA))
            tasks += [(m, meth, l, b) for l, b in grid if (round(l, 3), round(b, 4)) not in done]
    tasks.sort(key=lambda t: 0 if t[1] in ("mv_eac_norm", "entity_eac") else 1)
    print("pending", len(tasks), flush=True); t0 = time.time()

    def done(row, n, N):
        p = path(row["model"], row["method"])
        rows = pd.read_csv(p).to_dict("records") if p.exists() else []
        pd.DataFrame(rows + [row]).sort_values(["lambda", "beta"]).to_csv(p, index=False)
        print(f"[{n}/{N}] {row['model']}/{row['method']} l={row['lambda']} b={row['beta']} NDCG={row['NDCG@K']:.4f} "
              f"Ent={row['Entropy']:.4f} ERR={row['ERR@K']:.4f} TCI={row['TCI@K']:.4f} [{row['_elapsed_s']}s] "
              f"{(time.time()-t0)/60:.0f} min", flush=True)
    pool(_grid_task, tasks, workers, done)


def select(g):
    g = g.copy(); g["maut"] = maut(g, W)
    return g[g.maut >= g.maut.max() - TOL].sort_values(["beta", "lambda"]).iloc[0]


def stage_select():
    rows = []
    cols = ["lambda", "beta", "NDCG@K", "Entropy", "ERR@K", "TCI@K"]
    for m in MODELS:
        for meth in EAC + ["xquad"]:
            g = pd.read_csv(GR / f"val_grid_mind_{m}_{meth}.csv").replace({"beta": {"inf": np.inf}})
            g["beta"] = g.beta.astype(float)
            if meth == "xquad":
                rows.append({"model": m, "method": meth, **select(g)[cols].to_dict()}); continue
            r = select(g); top = g.beta[np.isfinite(g.beta)].max()
            rows.append({"model": m, "method": meth, **r[cols].to_dict(), "at_grid_top": bool(r.beta >= top)})
            g10 = g[g.beta <= 10.0]
            tc = g10[g10.beta == 0].assign(maut=maut(g10, W)[g10.beta == 0]).sort_values("maut", ascending=False).iloc[0]
            name = "tradcal_mv_norm" if meth == "mv_eac_norm" else "tradcal_" + meth.replace("_eac", "")
            rows.append({"model": m, "method": name, **tc[cols].to_dict()})
    df = pd.DataFrame(rows); df.to_csv(ROOT / "best_params_mind.csv", index=False)
    pd.set_option("display.width", 200); print(df.round(4).to_string(index=False))


def stage_test(workers):
    for d in ("per_impression", "lists"):
        (TE / d).mkdir(parents=True, exist_ok=True)
    init("test")
    bp = pd.read_csv(ROOT / "best_params_mind.csv")
    tasks = []
    for m in MODELS:
        tasks.append((m, "original", "original", 0.0, 0.0))
        for r in bp[bp.model == m].itertuples():
            meth = "mv_eac_norm" if r.method == "tradcal_mv_norm" else (
                r.method.replace("tradcal_", "") + "_eac" if r.method.startswith("tradcal_") else r.method)
            tasks.append((m, r.method, meth, float(r[3]), float(r.beta)))
    tasks = [t for t in tasks if not (TE / "per_impression" / f"{t[0]}_{t[1]}.parquet").exists()]
    summ = TE / "test_summary_mind.csv"
    rows = pd.read_csv(summ).to_dict("records") if summ.exists() else []

    def done(row, n, N):
        nonlocal rows
        rows = [x for x in rows if (x["model"], x["config"]) != (row["model"], row["config"])] + [row]
        pd.DataFrame(rows).to_csv(summ, index=False)
        print(f"[{n}/{N}] {row['model']}/{row['config']} NDCG={row['NDCG@K']:.4f} Ent={row['Entropy']:.4f} "
              f"ERR={row['ERR@K']:.4f} TCI={row['TCI@K']:.4f}", flush=True)
    pool(_test_task, tasks, workers, done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "select", "test"], required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    {"grid": lambda: stage_grid(a.workers), "select": stage_select, "test": lambda: stage_test(a.workers)}[a.stage]()
