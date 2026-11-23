#!/usr/bin/env python3
"""
Second revision, step 32 -- MIND-small statistics (paper Section 5.9, Supplementary Table S13), same protocol
as step 23 (main family):
ONE Holm / BH family over all rows below x 2 models x 5 metrics (NDCG, ILD, Entropy, ERR, TCI).

  A  EAC vs matched Trad-Cal: Category-, Topic-, Entity-EAC; MV-EAC-Norm vs Trad-Cal-MV-Norm
  D  MV-EAC-Norm vs Original;  T  Trad-Cal-MV-Norm vs Original
  G  MV-EAC-Norm vs Category-/Topic-/Entity-EAC
  K  MV-EAC-Norm vs xQuAD
Each row: impression-level d_z + Wilcoxon; user-level (per-user mean difference); |R_u| > K subset;
warm subset (users with a history / profile) and cold subset (no history).
Signed d: positive = better for the first method (ERR and TCI are minimized).
Output: $MVEAC_DATA_ROOT/mind/test/stats/
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
from mveac.evaluation.stats import cohens_d as dz, wilcoxon_p as wilc

ALPHA = C.ALPHA
METRICS = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
MIN_METRICS = {"ERR@K", "TCI@K"}


def holm_adj(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


def bh_adj(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)[::-1]; adj = np.empty(n); run = 1.0
    for r, i in enumerate(o):
        run = min(run, p[i] * n / (n - r)); adj[i] = run
    return adj


ROOT = C.DATA_ROOT / "mind"
TE, OUT = ROOT / "test", ROOT / "test" / "stats"
MODELS = ["most_popular", "nrms"]
NORM = "mv_eac_norm"
COMPS = ([("A", f"{v}_eac", f"tradcal_{v}") for v in ("category", "topic", "entity")]
         + [("A", NORM, "tradcal_mv_norm"), ("D", NORM, "original"), ("T", "tradcal_mv_norm", "original")]
         + [("G", NORM, f"{v}_eac") for v in ("category", "topic", "entity")] + [("K", NORM, "xquad")])


def per(model, cfg):
    return pd.read_parquet(TE / "per_impression" / f"{model}_{cfg}.parquet").set_index("impression_id").sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sub = pd.read_parquet(ROOT / "processed" / "test_sub.parquet")
    imp2user = sub.set_index("impression_id")["user_id"]
    hist = pickle.load(open(ROOT / "processed" / "history_test.pkl", "rb"))
    rows = []
    for model in MODELS:
        for fam, a, b in COMPS:
            A, B = per(model, a), per(model, b)
            idx = A.index.intersection(B.index); A, B = A.loc[idx], B.loc[idx]
            users = imp2user.reindex(idx).values
            gt = A["n_cand"].values > C.K
            warm = np.isin(users, list(hist))
            for m in METRICS:
                diff = (A[m].values - B[m].values).astype(float)
                sg = -1.0 if m in MIN_METRICS else 1.0
                ud = pd.DataFrame({"u": users, "d": diff}).groupby("u")["d"].mean().values
                rows.append({"family": fam, "model": model, "a": a, "b": b, "metric": m, "n_imp": len(diff),
                             "n_user": len(ud), "mean_a": A[m].mean(), "mean_b": B[m].mean(), "delta": diff.mean(),
                             "d_imp_signed": sg * dz(diff), "p_imp": wilc(diff),
                             "d_user_signed": sg * dz(ud), "p_user": wilc(ud),
                             "d_ru_gtK_signed": sg * dz(diff[gt]), "p_ru_gtK": wilc(diff[gt]),
                             "d_warm_signed": sg * dz(diff[warm]), "d_cold_signed": sg * dz(diff[~warm]),
                             "n_cold": int((~warm).sum())})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adj(df.p_imp); df["p_bh"] = bh_adj(df.p_imp)
    df["p_user_holm"] = holm_adj(df.p_user); df["p_ru_gtK_holm"] = holm_adj(df.p_ru_gtK)
    for c in ("holm", "bh", "user_holm", "ru_gtK_holm"):
        df[f"sig_{c}"] = df[f"p_{c}"] < ALPHA
    df.to_csv(OUT / "stats_mind_full.csv", index=False)
    lines = [f"MIND family: {len(df)} tests | Holm sig {df.sig_holm.sum()} | BH {df.sig_bh.sum()} | "
             f"user-level Holm {df.sig_user_holm.sum()} | |R_u|>K Holm {df.sig_ru_gtK_holm.sum()}",
             f"verdict changes at user level: {(df.sig_holm != df.sig_user_holm).sum()}", ""]
    piv = df.assign(cell=lambda d: d.d_imp_signed.round(2).astype(str) + np.where(d.sig_holm, "", "ns"))
    piv = piv.pivot_table(index=["model", "a", "b"], columns="metric", values="cell", aggfunc="first", sort=False)
    lines.append(piv[METRICS].to_string())
    txt = "\n".join(lines); (OUT / "stats_mind_summary.txt").write_text(txt); print(txt)


if __name__ == "__main__":
    main()
