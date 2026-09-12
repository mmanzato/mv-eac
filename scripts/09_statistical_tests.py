#!/usr/bin/env python3
"""
Step 9 of 10 -- statistical significance testing across every comparison the
paper makes.

For each base model, runs paired Wilcoxon signed-rank tests + Cohen's d on
the per-impression files written by step 8, across six comparison families:

    A  each single-view/MV-EAC vs. its matched Trad-Cal baseline   (RQ1)
    B  each MCF variant vs. its matched Trad-Cal baseline          (RQ1, if step 7 ran)
    C  each single-view EAC vs. its matched MCF variant            (RQ1, if step 7 ran)
    D  MV-EAC vs. the uncalibrated Original baseline                (RQ1)
    E  each RQ2 ablation variant vs. Full MV-EAC                    (RQ2)
    F  each RQ4 weight-sweep point vs. Full MV-EAC                  (RQ4)

Every test is run twice: once on impression-level paired differences, and
once after aggregating to one paired observation per user (the mean of that
user's own impression-level differences) to check that impression
non-independence is not driving the result. p-values are corrected for
multiplicity across the whole family of tests using both Holm-Bonferroni and
Benjamini-Hochberg.

Usage:
    python scripts/09_statistical_tests.py

Output (under $MVEAC_DATA_ROOT/results/):
    stats_full.csv       one row per (family, model, comparison, metric)
    stats_summary.txt     human-readable summary
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.evaluation.stats import benjamini_hochberg, compare, holm_bonferroni

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METRICS = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
MINIMIZE_METRICS = {"ERR@K", "TCI@K"}


def _load(model: str, method: str) -> pd.DataFrame | None:
    path = C.RESULTS_DIR / f"per_impression_{model}_{method}.csv"
    if not path.exists():
        return None
    keep = ["impression_id"] + METRICS
    return pd.read_csv(path, usecols=lambda c: c in keep).set_index("impression_id").sort_index()


def _comparisons(model: str) -> list[tuple[str, str, str, str]]:
    """(family, label, method_a, method_b) tuples for one base model."""
    out = []
    for view in ["category", "entity", "topic", "sentiment"]:
        out.append(("A_EAC_vs_TradCal", f"{view}_eac vs trad", f"{view}_eac", f"traditional_{view}_eac"))
    out.append(("A_EAC_vs_TradCal", "mv_eac vs trad", "mv_eac", "traditional_mv_eac"))
    out.append(("D_MV_vs_Original", "mv_eac vs original", "mv_eac", "original"))
    for name in C.ABLATION_CONFIGS:
        out.append(("E_Ablation_vs_MV", f"{name} vs mv_eac", f"ablation_{name}", "mv_eac"))

    pattern = str(C.RESULTS_DIR / f"per_impression_{model}_sweep_wtopic_*.csv")
    for path in sorted(glob.glob(pattern)):
        w_tag = Path(path).stem.split("sweep_wtopic_")[1]
        if abs(float(w_tag) - 1 / 3) < 1e-6:
            continue  # the uniform point IS mv_eac; comparing it to itself is meaningless
        out.append(("F_Sweep_vs_MV", f"sweep w_topic={w_tag} vs mv_eac", f"sweep_wtopic_{w_tag}", "mv_eac"))

    for view in ["category", "topic", "entity", "sentiment"]:
        if (C.RESULTS_DIR / f"per_impression_mcf_{model}_{view}.csv").exists():
            out.append(("B_MCF_vs_TradCal", f"mcf_{view} vs trad", f"mcf_{view}", f"traditional_{view}_eac"))
            out.append(("C_EAC_vs_MCF", f"{view}_eac vs mcf", f"{view}_eac", f"mcf_{view}"))
    return out


def main() -> None:
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet", columns=["impression_id", "user_id"])
    impression_to_user = test_sub.drop_duplicates("impression_id").set_index("impression_id")["user_id"]

    # MCF per-impression files live under a different filename prefix (written by
    # step 7); read them under the "mcf_{view}" method key so _comparisons() can
    # find them uniformly.
    for path in glob.glob(str(C.RESULTS_DIR / "per_impression_mcf_*.csv")):
        stem = Path(path).stem  # per_impression_mcf_{model}_{view}
        model_view = stem[len("per_impression_mcf_"):]
        model = next((m for m in C.MODELS if model_view.startswith(m + "_")), None)
        if model is None:
            continue
        view = model_view[len(model) + 1:]
        alias = C.RESULTS_DIR / f"per_impression_{model}_mcf_{view}.csv"
        if not alias.exists():
            alias.write_bytes(Path(path).read_bytes())

    rows = []
    for model in C.MODELS:
        for family, label, method_a, method_b in _comparisons(model):
            df_a, df_b = _load(model, method_a), _load(model, method_b)
            if df_a is None or df_b is None:
                continue
            common = df_a.index.intersection(df_b.index)
            users = impression_to_user.reindex(common)
            for metric in METRICS:
                if metric not in df_a.columns or metric not in df_b.columns:
                    continue
                result = compare(
                    df_a.loc[common, metric], df_b.loc[common, metric], users,
                    minimize=metric in MINIMIZE_METRICS,
                )
                rows.append({"family": family, "model": model, "comparison": label, "metric": metric, **result})

    stats_df = pd.DataFrame(rows)
    stats_df["significant_uncorrected"] = stats_df["p_impression"] < C.ALPHA
    stats_df["significant_holm"] = holm_bonferroni(stats_df["p_impression"].fillna(1).values, C.ALPHA)
    stats_df["significant_bh"] = benjamini_hochberg(stats_df["p_impression"].fillna(1).values, C.ALPHA)
    stats_df["significant_user_holm"] = holm_bonferroni(stats_df["p_user"].fillna(1).values, C.ALPHA)
    stats_df["practically_meaningful"] = stats_df["d_impression"].abs() >= C.EFFECT_SIZE_THRESHOLDS["negligible"]

    stats_df.to_csv(C.RESULTS_DIR / "stats_full.csv", index=False)

    lines = [
        "=" * 78, "Statistical test summary", "=" * 78,
        f"Total tests: {len(stats_df)}",
        f"Significant: uncorrected={stats_df.significant_uncorrected.sum()} "
        f"Holm={stats_df.significant_holm.sum()} BH={stats_df.significant_bh.sum()}",
        f"Significant (user-clustered, Holm): {stats_df.significant_user_holm.sum()}", "",
    ]
    for family in stats_df.family.unique():
        subset = stats_df[stats_df.family == family]
        lines.append(
            f"  {family:22s} Holm {subset.significant_holm.sum():>3}/{len(subset):<3}  "
            f"|d|>=0.10: {subset.practically_meaningful.sum():>3}"
        )
    (C.RESULTS_DIR / "stats_summary.txt").write_text("\n".join(lines))
    log.info("\n%s", "\n".join(lines))
    log.info("Full results -> %s", C.RESULTS_DIR / "stats_full.csv")


if __name__ == "__main__":
    main()
