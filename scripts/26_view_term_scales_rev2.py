#!/usr/bin/env python3
"""
Second revision, step 26 -- per-view effective influence of the MV-EAC terms along the greedy path (paper Section 5.6,
Supplementary Section S7), for the normalized MV-EAC (revised) and the un-normalized MV-EAC_nom, each at its
step-20 selection. Same sample as step 16: 3,000 validation impressions
with |R_u|>K per base model (rng seed 0). For each greedy step with >1 active candidate we record the
spread (sd over active candidates) of each weighted term and whether removing one view's KL+UCB terms
changes the selected article ("flip").
Output: results/rev2/term_scales_rev2.csv
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
from mveac.calibration.eac import _build_article_matrix, _kl_all, _minmax_active as _mm
from mveac.calibration.traditional import minmax_normalize as _minmax_normalize


def _kl_vec(p, Q):
    return _kl_all(p, Q, C.KL_EPSILON)


VIEWS = ["category", "entity", "topic"]; W = {v: 1 / 3 for v in VIEWS}
vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
pall = pickle.load(open(C.SEM_DIR / "user_profiles.pkl", "rb"))
val = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
imp2u = dict(zip(val.impression_id.astype(int), val.user_id.astype(int)))
best = pd.read_csv(C.RESULTS_DIR / "rev2" / "best_params_rev2.csv")
idx = {v: {x: i for i, x in enumerate(vocabs[v])} for v in VIEWS}
rows = []
for model in C.MODELS:
    sc_all = pd.read_parquet(C.score_path(model, "val"))
    sizes = sc_all.groupby("impression_id").size()
    ids = np.random.default_rng(0).choice(sizes[sizes > C.K].index.values, 3000, replace=False)
    sc = sc_all[sc_all.impression_id.isin(ids)]
    for form, meth in (("norm", "mv_eac_norm"), ("nom", "mv_eac")):
        lam, beta = best[(best.model == model) & (best.config == meth)][["lambda", "beta"]].iloc[0]
        norm = form == "norm"
        for iid, g in sc.groupby("impression_id"):
            cands = [(int(a), float(s)) for a, s in g.sort_values("rank")[["article_id", "score"]].itertuples(index=False)]
            uid = imp2u[int(iid)]
            prof = {v: (pall[v][uid] if uid in pall[v] else np.ones(len(vocabs[v])) / len(vocabs[v])) for v in VIEWS}
            cid = [c for c, _ in cands]; n = len(cid)
            A = {v: _build_article_matrix(cid, maps[v], idx[v], len(vocabs[v])).astype(float) for v in VIEWS}
            rel = np.array(_minmax_normalize([s for _, s in cands]))
            L = {v: np.zeros(len(vocabs[v])) for v in VIEWS}; N = {v: np.zeros(len(vocabs[v])) for v in VIEWS}
            ncat = {v: A[v].sum(1) for v in VIEWS}
            active = np.ones(n, bool); T = 0
            for step in range(min(C.K, n)):
                kl, ucb = {}, {}
                for v in VIEWS:
                    Q = L[v][None, :] + A[v]; tot = Q.sum(1, keepdims=True); tot = np.where(tot > 0, tot, 1.0)
                    kl[v] = _kl_vec(prof[v], Q / tot)
                    if beta > 0 and T > 0:
                        upc = (1 - prof[v]) * np.sqrt(np.log(T + 1) / (N[v] + 1))
                        ucb[v] = np.where(ncat[v] > 0, (A[v] @ upc) / np.where(ncat[v] > 0, ncat[v], 1), 0.0)
                    else:
                        ucb[v] = np.zeros(n)
                    if norm:
                        kl[v] = _mm(kl[v], active); ucb[v] = _mm(ucb[v], active)
                terms = {"rel": (1 - lam) * rel}
                for v in VIEWS:
                    terms[f"kl_{v}"] = -lam * W[v] * kl[v]
                    terms[f"ucb_{v}"] = beta * W[v] * ucb[v]
                total = sum(terms.values())
                a = np.where(active)[0]
                best_i = a[np.argmax(total[a])]
                if len(a) > 1:
                    rec = {"model": model, "form": form, "step": step, "n_active": len(a)}
                    for k_, x in terms.items():
                        rec["sd_" + k_] = float(np.std(x[a]))
                    for v in VIEWS:
                        rec["sd_view_" + v] = float(np.std((terms[f"kl_{v}"] + terms[f"ucb_{v}"])[a]))
                        alt = total - terms[f"kl_{v}"] - terms[f"ucb_{v}"]
                        rec["flip_" + v] = int(a[np.argmax(alt[a])] != best_i)
                    rows.append(rec)
                for v in VIEWS:
                    L[v] += A[v][best_i]; N[v] += A[v][best_i]
                active[best_i] = False; T += 1
        print(model, form, lam, beta, flush=True)
df = pd.DataFrame(rows)
df.to_csv(C.RESULTS_DIR / "rev2" / "term_scales_rev2.csv", index=False)
pd.set_option("display.width", 200)
print(df.groupby(["form", "model"])[[c for c in df.columns if c.startswith("flip_")]].mean().round(3).to_string())
print(df.groupby(["form", "model"])[[c for c in df.columns if c.startswith("sd_view_")]].median().round(4).to_string())
