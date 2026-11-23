#!/usr/bin/env python3
"""
Second revision, step 23 -- statistical families of the revised paper.

Main family (one Holm / Benjamini-Hochberg family, 735 tests = 49 comparisons x 5 metrics x 3 base
models, plus the NRMS-run comparisons). Code family names map to the paper's protocol letters
(Section 4.7) as given in PAPER_PROTOCOL below. Each row: impression-level d_z + Wilcoxon, user-level
(per-user mean difference), |R_u| > K subset, and users-with-history subset.

Secondary family (separate Holm; paper Section 4.5.4 / 5.3.7): the list-derived metrics of step 22
(@5, position-weighted, per-view C_KL) for the headline comparisons, on all impressions, |R_u| > K,
users with history and users without history.

Inputs: results/rev2/per_impression (step 21), results/per_impression_mcf_*.csv (step 07),
results/rev2/derived/per_impression (step 22).
Output (results/rev2/stats): stats_rev2_full.csv, stats_rev2_secondary.csv, pooled_rev2.csv,
ru_gtK_means_rev2.csv, stats_rev2_summary.txt
"""
from __future__ import annotations
import argparse, sys
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
from mveac.evaluation.stats import cohens_d as dz, wilcoxon_p as wilc  # with the 1e-9 tie tolerance

ALPHA = 0.05
METRICS = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
MIN_METRICS = {"ERR@K", "TCI@K"}
TEST = C.RESULTS_DIR / "rev2"
MCF_DIR = C.RESULTS_DIR
MODELS = list(C.MODELS)
# code family -> paper protocol letter (Section 4.7)
PAPER_PROTOCOL = {"A_EAC_vs_TradCal": "A", "B_MCF_vs_TradCal": "B", "C_EAC_vs_MCF": "C", "D_MV_vs_Original": "D",
                  "T_TradCalMV_vs_Original": "E", "E_Ablation_vs_MV": "F", "F_Sweep_vs_MV": "G",
                  "G_MV_vs_SingleView": "H", "H_SentimentAddBack": "I", "I_NRMS_Variants": "J",
                  "J_NRMS_Variant_Ablation": "K", "K_MV_vs_Baselines": "L", "L_Norm_vs_Nominal": "M",
                  "M_UCB_Components": "N", "N_D1_EntityControls": "O", "O_Epsilon": "P"}
VARIANTS = ["nrms_seed1", "nrms_seed2", "nrms_large1M"]
NORM = "mv_eac_norm"


@lru_cache(maxsize=None)
def per(model: str, cfg: str) -> pd.DataFrame | None:
    if cfg.startswith("mcf_"):
        p = MCF_DIR / f"per_impression_{cfg.replace('mcf_', 'mcf_' + model + '_')}.csv"  # per_impression_mcf_<model>_<view>.csv
        if not p.exists():
            return None
        df = pd.read_csv(p, usecols=["impression_id"] + METRICS)
        df["n_cand"] = per(model, "original")["n_cand"].reindex(df.impression_id).values
        return df.set_index("impression_id").sort_index()
    p = TEST / "per_impression" / f"{model}_{cfg}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=["impression_id", "n_cand"] + METRICS).set_index("impression_id").sort_index()


def comparisons(model: str):
    out = [("A_EAC_vs_TradCal", f"{v}_eac vs tradcal", f"{v}_eac", f"tradcal_{v}")
           for v in ["topic", "entity", "category", "sentiment"]]
    out += [("A_EAC_vs_TradCal", "mv_eac_norm vs tradcal_mv_norm", NORM, "tradcal_mv_norm"),
            ("A_EAC_vs_TradCal", "mv_eac (nominal) vs tradcal_mv", "mv_eac", "tradcal_mv")]
    for v in ["category", "topic", "entity", "sentiment"]:
        out += [("B_MCF_vs_TradCal", f"mcf_{v} vs tradcal", f"mcf_{v}", f"tradcal_{v}"),
                ("C_EAC_vs_MCF", f"{v}_eac vs mcf", f"{v}_eac", f"mcf_{v}")]
    out += [("D_MV_vs_Original", "mv_eac_norm vs original", NORM, "original"),
            ("T_TradCalMV_vs_Original", "tradcal_mv_norm vs original", "tradcal_mv_norm", "original")]
    out += [("E_Ablation_vs_MV", f"{a} vs mv_eac_norm", f"ablation_{a}", NORM)
            for a in ["no_category", "no_entity", "no_topic"]]
    out += [("F_Sweep_vs_MV", f"sweep w_topic={t} vs mv_eac_norm", f"sweep_wtopic_{t}", NORM)
            for t in ["0.15", "0.50", "0.70", "0.85", "1.00"]]
    out += [("G_MV_vs_SingleView", f"mv_eac_norm vs {v}_eac", NORM, f"{v}_eac") for v in ["topic", "entity", "category"]]
    out += [("H_SentimentAddBack", "plus_sentiment vs mv_eac_norm", "plus_sentiment", NORM)]
    out += [("K_MV_vs_Baselines", f"mv_eac_norm vs {b}", NORM, b) for b in ["mmr", "xquad", "dpp", "dedup"]]
    out += [("L_Norm_vs_Nominal", "mv_eac_norm vs mv_eac", NORM, "mv_eac")]
    for meth, full in [("mv_eac_norm", NORM), ("topic_eac", "topic_eac"), ("entity_eac", "entity_eac")]:
        for comp in ["hist", "list"]:
            out.append(("M_UCB_Components", f"{meth} {comp}-only vs full", f"c1_{meth}_{comp}", full))
    out += [("N_D1_EntityControls", "no_entity_fixed vs mv_eac_norm", "d1_no_entity_fixed", NORM),
            ("N_D1_EntityControls", "cat_topic_tuned vs mv_eac_norm", "d1_cat_topic_tuned", NORM)]
    out += [("O_Epsilon", f"eps={e}: mv_eac_norm vs tradcal_mv_norm", f"eps{e}_mv_eac_norm", f"eps{e}_tradcal_mv_norm")
            for e in ["1e-04", "1e-06", "1e-08"]]
    return out


def variant_comparisons():
    out = []
    for v in VARIANTS:
        out.append((v, ("I_NRMS_Variants", "mv_eac_norm vs tradcal_mv_norm", NORM, "tradcal_mv_norm")))
        for a in ["no_category", "no_entity", "no_topic"]:
            out.append((v, ("J_NRMS_Variant_Ablation", f"{a} vs mv_eac_norm", f"ablation_{a}", NORM)))
        out.append((v, ("J_NRMS_Variant_Ablation", "plus_sentiment vs mv_eac_norm", "plus_sentiment", NORM)))
    return out


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


def cold_set():
    sc = pd.read_parquet(C.score_path("nrms", "test"), columns=["impression_id", "score"])
    a = sc.groupby("impression_id")["score"].apply(lambda x: x.isna().all())
    return set(a[a].index.astype(int))


def main_family(imp2user, cold):
    rows = []
    jobs = [(m, c) for m in MODELS for c in comparisons(m)] + variant_comparisons()
    for model, (fam, label, ma, mb) in jobs:
        A, B = per(model, ma), per(model, mb)
        if A is None or B is None:
            print("MISSING", model, ma if A is None else mb); continue
        idx = A.index.intersection(B.index)
        A, B = A.loc[idx], B.loc[idx]
        users = imp2user.reindex(idx).values
        gt = (A["n_cand"].values > C.K)
        warm = ~np.isin(idx.values, list(cold))
        for m in METRICS:
            diff = (A[m].values - B[m].values).astype(float)
            sgn = -1.0 if m in MIN_METRICS else 1.0
            ud = pd.DataFrame({"u": users, "d": diff}).groupby("u")["d"].mean().values
            rows.append({"family": fam, "model": model, "comparison": label, "a": ma, "b": mb, "metric": m,
                         "n_imp": len(diff), "n_user": len(ud), "mean_a": A[m].mean(), "mean_b": B[m].mean(),
                         "delta": diff.mean(), "d_imp": dz(diff), "d_imp_signed": sgn * dz(diff), "p_imp": wilc(diff),
                         "d_user": dz(ud), "d_user_signed": sgn * dz(ud), "p_user": wilc(ud),
                         "d_imp_ru_gtK": dz(diff[gt]), "d_imp_ru_gtK_signed": sgn * dz(diff[gt]),
                         "p_imp_ru_gtK": wilc(diff[gt]), "d_imp_warm": dz(diff[warm])})
    df = pd.DataFrame(rows)
    df.insert(1, "paper_protocol", df.family.map(PAPER_PROTOCOL))
    df["p_holm"] = holm_adj(df.p_imp.fillna(1)); df["p_bh"] = bh_adj(df.p_imp.fillna(1))
    df["p_user_holm"] = holm_adj(df.p_user.fillna(1))
    df["p_ru_gtK_holm"] = holm_adj(df.p_imp_ru_gtK.fillna(1))
    df["sig_uncorr"] = df.p_imp < ALPHA
    df["sig_holm"] = df.p_holm < ALPHA; df["sig_bh"] = df.p_bh < ALPHA
    df["sig_user_holm"] = df.p_user_holm < ALPHA
    df["real_imp"] = df.d_imp.abs() >= 0.10; df["real_user"] = df.d_user.abs() >= 0.10
    return df


SEC_METRICS = ["NDCG10", "ILD10", "ENT10", "ERR10", "TCI10", "NDCG5", "ILD5", "ENT5", "ERR5", "TCI5",
               "ILDpw", "ENTpw", "ERRpw", "TCIpw", "CKL_cat", "CKL_top", "CKL_ent", "CKL_sen"]
SEC_COMPS = [("A", NORM, "tradcal_mv_norm"), ("D", NORM, "original"), ("T", "tradcal_mv_norm", "original"),
             ("G", NORM, "topic_eac")] + [("K", NORM, b) for b in ["mmr", "xquad", "dpp", "dedup"]]


def secondary(derived: Path):
    rows = []
    for model in MODELS:
        cache = {}
        def get(m):
            if m not in cache:
                cache[m] = pd.read_parquet(derived / "per_impression" / f"{model}_{m}.parquet").set_index("impression_id").sort_index()
            return cache[m]
        for fam, ma, mb in SEC_COMPS:
            A, B = get(ma), get(mb)
            assert A.index.equals(B.index)
            subsets = {"all": np.ones(len(A), bool), "ru_gtK": (A.n_cand > C.K).values,
                       "warm": (~A.nrms_nan).values, "cold": A.nrms_nan.values}
            for sub, mask in subsets.items():
                for m in SEC_METRICS:
                    diff = (A[m].values - B[m].values)[mask]
                    sgn = -1.0 if m.startswith(("ERR", "TCI", "CKL")) else 1.0
                    rows.append(dict(family=fam, model=model, comparison=f"{ma} vs {mb}", subset=sub, metric=m,
                                     n=int(mask.sum()), mean_a=np.nanmean(A[m].values[mask]),
                                     mean_b=np.nanmean(B[m].values[mask]), delta=np.nanmean(diff),
                                     d=dz(diff), d_signed=sgn * dz(diff), p=wilc(diff)))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adj(df.p.fillna(1)); df["sig_holm"] = df.p_holm < ALPHA
    return df


def pooled(df):
    prow = []
    for fam in ["E_Ablation_vs_MV", "F_Sweep_vs_MV", "H_SentimentAddBack", "K_MV_vs_Baselines", "M_UCB_Components",
                "N_D1_EntityControls"]:
        for (label, ma, mb), _ in df[df.family == fam].groupby(["comparison", "a", "b"]):
            for m in METRICS:
                x = []
                for model in MODELS:
                    A, B = per(model, ma), per(model, mb)
                    i = A.index.intersection(B.index)
                    x.append(A.loc[i, m].values - B.loc[i, m].values)
                x = np.concatenate(x)
                s = df[(df.family == fam) & (df.comparison == label) & (df.metric == m) & df.model.isin(MODELS)]
                prow.append({"family": fam, "comparison": label, "metric": m, "n": len(x), "delta": x.mean(),
                             "d_pooled": dz(x), "d_min": s.d_imp.min(), "d_max": s.d_imp.max(),
                             "all_models_sig_holm": bool(s.sig_holm.all())})
    return pd.DataFrame(prow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TEST / "stats"))
    ap.add_argument("--derived", default=str(TEST / "derived"))
    ap.add_argument("--skip-secondary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imp2user = (pd.read_parquet(C.PROC_DIR / "test_sub.parquet", columns=["impression_id", "user_id"])
                .drop_duplicates("impression_id").set_index("impression_id")["user_id"])
    df = main_family(imp2user, cold_set())
    df.to_csv(out / "stats_rev2_full.csv", index=False)
    pooled(df).to_csv(out / "pooled_rev2.csv", index=False)
    rr = []
    for p in sorted((TEST / "per_impression").glob("*.parquet")):
        model = next(m for m in VARIANTS + MODELS if p.stem.startswith(m + "_"))
        cfg = p.stem[len(model) + 1:]
        x = per(model, cfg); g = x[x.n_cand > C.K]
        rr.append({"model": model, "config": cfg, "n": len(g), **{m: g[m].mean() for m in METRICS}})
    pd.DataFrame(rr).to_csv(out / "ru_gtK_means_rev2.csv", index=False)
    if not a.skip_secondary:
        secondary(Path(a.derived)).to_csv(out / "stats_rev2_secondary.csv", index=False)

    L = [f"Main family: {len(df)} tests | sig uncorr {df.sig_uncorr.sum()}  Holm {df.sig_holm.sum()}  BH {df.sig_bh.sum()}"
         f"  user-level Holm {df.sig_user_holm.sum()}", ""]
    for fam, s in df.groupby("family", sort=False):
        L.append(f"  {fam:26s} Holm {s.sig_holm.sum():>3}/{len(s):<3} |d_imp|>=.10 {s.real_imp.sum():>3} "
                 f"|d_user|>=.10 {s.real_user.sum():>3}")
    lost = df[df.sig_uncorr & ~df.sig_holm]
    L.append(f"\nLose significance under Holm: {len(lost)} (max |d| {lost.d_imp.abs().max() if len(lost) else float('nan'):.3f})")
    flip = df[df.sig_holm != df.sig_user_holm]
    L.append(f"Verdict changes at user level: {len(flip)}; max |d_imp| "
             f"{flip.d_imp.abs().max() if len(flip) else float('nan'):.3f}")
    nr = df[df.sig_holm & ~df.real_imp]
    L.append(f"Holm-significant but |d| < 0.10: {len(nr)}")
    (out / "stats_rev2_summary.txt").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
