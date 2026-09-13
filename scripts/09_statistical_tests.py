#!/usr/bin/env python3
"""
Step 9 of 10 -- statistical significance testing across every comparison the
paper makes.

For each base model, runs paired Wilcoxon signed-rank tests + Cohen's d_z on
the per-impression files written by step 8, across nine comparison families
that together form ONE Holm-Bonferroni / Benjamini-Hochberg family (405 tests):

    A  each single-view EAC and MV-EAC vs. its matched Trad-Cal baseline   (RQ1)   75
    B  each MCF variant vs. its matched Trad-Cal baseline                  (RQ1)   60
    C  each single-view EAC vs. its matched MCF variant                    (RQ1)   60
    D  MV-EAC vs. the uncalibrated Original baseline                       (RQ1)   15
    E  each RQ2 ablation variant vs. Full MV-EAC                           (RQ2)   45
    F  each RQ4 weight-sweep point vs. Full MV-EAC                         (RQ4)   75
       (w_topic = 0.00 is the same configuration as the No-Topic ablation and
        is not tested twice)
    G  MV-EAC vs. each retained single-view EAC                            (RQ3)   45
    H  sentiment add-back (4 views at MV-EAC's lambda/beta) vs. MV-EAC     (RQ2)   15
    I  MV-EAC vs. Trad-Cal-MV on the three extra NRMS training runs        (RQ1)   15

Families B/C need step 7; family I needs ``08_evaluate_test.py --nrms-tag``.

Every test is run twice: on impression-level paired differences, and after
aggregating to one paired observation per user (mean of that user's differences).
Effect sizes use the RAW difference (method_a - method_b), as in the paper's tables.

Also written:
    pooled.csv        pooled (all-models) descriptive d for families E, F, H, with a flag
                      that is True only if all three per-model tests are Holm-significant
    ru_gtK_means.csv  every configuration's metric means restricted to |R_u| > K

Usage:
    python scripts/09_statistical_tests.py
"""
from __future__ import annotations

import glob
import logging
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.evaluation.stats import benjamini_hochberg, cohens_d, compare, holm_bonferroni

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METRICS = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
MINIMIZE_METRICS = {"ERR@K", "TCI@K"}
NRMS_VARIANT_TAGS = ["seed1", "seed2", "large1M"]


def _base_model(model: str) -> str:
    return model.split("_seed")[0].split("_large")[0]


@lru_cache(maxsize=None)
def _candidate_counts(model: str) -> pd.Series:
    df = pd.read_csv(C.RESULTS_DIR / f"per_impression_{_base_model(model)}_original.csv",
                     usecols=["impression_id", "n_candidates"])
    return df.set_index("impression_id")["n_candidates"]


@lru_cache(maxsize=None)
def _load(model: str, method: str) -> pd.DataFrame | None:
    path = C.RESULTS_DIR / f"per_impression_{model}_{method}.csv"
    if not path.exists():
        return None
    keep = ["impression_id", "n_candidates"] + METRICS
    df = pd.read_csv(path, usecols=lambda c: c in keep).set_index("impression_id").sort_index()
    if "n_candidates" not in df.columns:  # MCF files: candidate sets are model-invariant
        df["n_candidates"] = _candidate_counts(model).reindex(df.index).values
    return df


def _comparisons(model: str) -> list[tuple[str, str, str, str]]:
    """(family, label, method_a, method_b) tuples for one base model."""
    out = []
    for view in ["topic", "entity", "category", "sentiment"]:
        out.append(("A_EAC_vs_TradCal", f"{view}_eac vs trad", f"{view}_eac", f"traditional_{view}_eac"))
    out.append(("A_EAC_vs_TradCal", "mv_eac vs trad", "mv_eac", "traditional_mv_eac"))
    out.append(("D_MV_vs_Original", "mv_eac vs original", "mv_eac", "original"))
    for name in C.ABLATION_CONFIGS:
        out.append(("E_Ablation_vs_MV", f"{name} vs mv_eac", f"ablation_{name}", "mv_eac"))
    for w_topic in C.WEIGHT_SWEEP:
        if w_topic == 0.0:
            continue  # identical to ablation_no_topic
        tag = f"{w_topic:.2f}"
        out.append(("F_Sweep_vs_MV", f"sweep w_topic={tag} vs mv_eac", f"sweep_wtopic_{tag}", "mv_eac"))
    for view in ["topic", "entity", "category"]:
        out.append(("G_MV_vs_SingleView", f"mv_eac vs {view}_eac", "mv_eac", f"{view}_eac"))
    out.append(("H_SentimentAddBack", "mv4 vs mv_eac", "mv4_at_mv3params", "mv_eac"))
    for view in ["category", "topic", "entity", "sentiment"]:
        out.append(("B_MCF_vs_TradCal", f"mcf_{view} vs trad", f"mcf_{view}", f"traditional_{view}_eac"))
        out.append(("C_EAC_vs_MCF", f"{view}_eac vs mcf", f"{view}_eac", f"mcf_{view}"))
    return out


def main() -> None:
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet", columns=["impression_id", "user_id"])
    impression_to_user = test_sub.drop_duplicates("impression_id").set_index("impression_id")["user_id"]

    # MCF per-impression files are written by step 7 as per_impression_mcf_{model}_{view}.csv;
    # alias them to per_impression_{model}_mcf_{view}.csv so _comparisons() finds them uniformly.
    for path in glob.glob(str(C.RESULTS_DIR / "per_impression_mcf_*.csv")):
        model_view = Path(path).stem[len("per_impression_mcf_"):]
        model = next((m for m in C.MODELS if model_view.startswith(m + "_")), None)
        if model is None:
            continue
        alias = C.RESULTS_DIR / f"per_impression_{model}_mcf_{model_view[len(model) + 1:]}.csv"
        if not alias.exists():
            alias.write_bytes(Path(path).read_bytes())

    jobs = [(model, c) for model in C.MODELS for c in _comparisons(model)]
    jobs += [(f"nrms_{tag}", ("I_NRMS_Variants", "mv_eac vs trad", "mv_eac", "traditional_mv_eac"))
             for tag in NRMS_VARIANT_TAGS]

    rows = []
    for model, (family, label, method_a, method_b) in jobs:
        df_a, df_b = _load(model, method_a), _load(model, method_b)
        if df_a is None or df_b is None:
            log.warning("skipping %s / %s (missing per-impression file)", model, label)
            continue
        common = df_a.index.intersection(df_b.index)
        users = impression_to_user.reindex(common)
        gt_k = df_a.loc[common, "n_candidates"].values > C.K
        for metric in METRICS:
            result = compare(df_a.loc[common, metric], df_b.loc[common, metric], users,
                             minimize=metric in MINIMIZE_METRICS)
            diff = (df_a.loc[common, metric].values - df_b.loc[common, metric].values).astype(float)
            rows.append({"family": family, "model": model, "comparison": label, "metric": metric,
                         **result, "d_impression_ru_gtK": cohens_d(diff[gt_k])})

    stats_df = pd.DataFrame(rows)
    stats_df["significant_uncorrected"] = stats_df["p_impression"] < C.ALPHA
    stats_df["significant_holm"] = holm_bonferroni(stats_df["p_impression"].fillna(1).values, C.ALPHA)
    stats_df["significant_bh"] = benjamini_hochberg(stats_df["p_impression"].fillna(1).values, C.ALPHA)
    stats_df["significant_user_holm"] = holm_bonferroni(stats_df["p_user"].fillna(1).values, C.ALPHA)
    negligible = C.EFFECT_SIZE_THRESHOLDS["negligible"]
    stats_df["practically_meaningful"] = stats_df["d_impression"].abs() >= negligible
    stats_df["practically_meaningful_user"] = stats_df["d_user"].abs() >= negligible
    stats_df.to_csv(C.RESULTS_DIR / "stats_full.csv", index=False)

    # ---- pooled descriptive effects (ablation / sweep / add-back tables) -------------
    pooled = []
    for family in ["E_Ablation_vs_MV", "F_Sweep_vs_MV", "H_SentimentAddBack"]:
        for label in stats_df[stats_df.family == family].comparison.unique():
            method_a, method_b = next((a, b) for f, l, a, b in _comparisons(C.MODELS[0]) if l == label)
            for metric in METRICS:
                diffs = []
                for model in C.MODELS:
                    df_a, df_b = _load(model, method_a), _load(model, method_b)
                    common = df_a.index.intersection(df_b.index)
                    diffs.append(df_a.loc[common, metric].values - df_b.loc[common, metric].values)
                x = np.concatenate(diffs)
                sub = stats_df[(stats_df.family == family) & (stats_df.comparison == label)
                               & (stats_df.metric == metric)]
                pooled.append({"family": family, "comparison": label, "metric": metric, "n": len(x),
                               "delta": float(x.mean()), "d_pooled": cohens_d(x),
                               "d_min": sub.d_impression.min(), "d_max": sub.d_impression.max(),
                               "all_models_sig_holm": bool(sub.significant_holm.all())})
    pd.DataFrame(pooled).to_csv(C.RESULTS_DIR / "pooled.csv", index=False)

    # ---- means restricted to |R_u| > K ------------------------------------------------
    ru_rows = []
    for model in C.MODELS:
        for path in sorted(glob.glob(str(C.RESULTS_DIR / f"per_impression_{model}_*.csv"))):
            method = Path(path).stem[len(f"per_impression_{model}_"):]
            if method.startswith(("seed", "large")):
                continue
            df = _load(model, method)
            sel = df[df.n_candidates > C.K]
            ru_rows.append({"model": model, "method": method, "n": len(sel), **{m: sel[m].mean() for m in METRICS}})
    pd.DataFrame(ru_rows).to_csv(C.RESULTS_DIR / "ru_gtK_means.csv", index=False)

    lost = stats_df[stats_df.significant_uncorrected & ~stats_df.significant_holm]
    lines = [
        "=" * 78, "Statistical test summary", "=" * 78,
        f"Total tests: {len(stats_df)}",
        f"Significant: uncorrected={stats_df.significant_uncorrected.sum()} "
        f"Holm={stats_df.significant_holm.sum()} BH={stats_df.significant_bh.sum()}",
        f"Significant (user-clustered, Holm): {stats_df.significant_user_holm.sum()}",
        f"Largest |d| among tests that lose significance under Holm: "
        f"{lost.d_impression.abs().max() if len(lost) else float('nan'):.3f}", "",
    ]
    for family in stats_df.family.unique():
        subset = stats_df[stats_df.family == family]
        lines.append(f"  {family:22s} Holm {subset.significant_holm.sum():>3}/{len(subset):<3}  "
                     f"|d|>=0.10: {subset.practically_meaningful.sum():>3}")
    (C.RESULTS_DIR / "stats_summary.txt").write_text("\n".join(lines))
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
