#!/usr/bin/env python3
"""
Step 10 of 10 -- the four result figures used in the paper.

    fig_method_comparison.pdf     Original / Trad-Cal-MV / MV-EAC / Topic-EAC /
                                   Entity-EAC across all 5 metrics, one panel per
                                   metric, grouped by base model.
    fig_lambda_beta_heatmap.pdf   the full (lambda, beta) validation grid for
                                   MV-EAC, one panel per model, showing the beta
                                   plateau past ~0.9.
    fig_ablation.pdf              Cohen's d of each RQ2 ablation variant vs.
                                   Full MV-EAC, per metric (diverging, signed).
    fig_weight_sweep.pdf          the RQ4 continuous topic-weight sweep:
                                   NDCG/ILD/Entropy/TCI as a function of w_topic.

Usage:
    python scripts/10_make_figures.py

Output (under $MVEAC_DATA_ROOT/figures/): the four PDFs above.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C

# Categorical palette validated for colorblind-safety (see the dataviz notes in
# docs/pipeline.md); one fixed color per base model everywhere in the paper.
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
MODEL_COLOR = {"most_popular": C_BLUE, "itemknn": C_ORANGE, "nrms": C_AQUA}
MODEL_LABEL = {"most_popular": "Pop", "itemknn": "KNN", "nrms": "NRMS"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "svg.fonttype": "none",
})


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x")


def fig_method_comparison() -> None:
    summary = pd.read_csv(C.RESULTS_DIR / "test_summary.csv")
    methods = ["original", "traditional_mv_eac", "mv_eac", "topic_eac", "entity_eac"]
    labels = ["Original", "Trad-Cal-MV", "MV-EAC", "Topic-EAC", "Entity-EAC"]
    metrics = [("NDCG@K", "NDCG@10", True), ("ILD@K", "ILD@10", True), ("Entropy", "Entropy@10", True),
               ("ERR@K", "ERR@10", False), ("TCI@K", "TCI@10", False)]

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.2), constrained_layout=True)
    bar_width = 0.8 / len(C.MODELS)
    for ax, (column, title, higher_is_better) in zip(axes, metrics):
        for i, model in enumerate(C.MODELS):
            values = []
            for method in methods:
                row = summary[(summary.model == model) & (summary.method == method)]
                values.append(float(row[column].iloc[0]) if len(row) else np.nan)
            x = np.arange(len(methods)) + (i - (len(C.MODELS) - 1) / 2) * bar_width
            ax.bar(x, values, width=bar_width * 0.92, color=MODEL_COLOR[model], label=MODEL_LABEL[model])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.6)
        ax.set_title(f"{title} {'↑' if higher_is_better else '↓'}", fontsize=8.5)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        _style(ax)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=8)
    fig.savefig(C.FIGURES_DIR / "fig_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_method_comparison.pdf")


def fig_lambda_beta_heatmap() -> None:
    fig, axes = plt.subplots(1, len(C.MODELS), figsize=(7.0, 2.5), constrained_layout=True)
    image = None
    for ax, model in zip(axes, C.MODELS):
        grid = pd.read_csv(C.RESULTS_DIR / f"val_grid_{model}_mv_eac.csv")
        lambdas, betas = sorted(grid["lambda"].unique()), sorted(grid["beta"].unique())
        matrix = np.full((len(lambdas), len(betas)), np.nan)
        for i, lam in enumerate(lambdas):
            for j, beta in enumerate(betas):
                row = grid[(grid["lambda"].round(3) == round(lam, 3)) & (grid["beta"].round(3) == round(beta, 3))]
                if len(row):
                    matrix[i, j] = row["NDCG@K"].iloc[0]
        image = ax.imshow(matrix, aspect="auto", cmap="Blues", origin="lower")
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"{b:g}" for b in betas], rotation=60, fontsize=6.2)
        ax.set_yticks(range(len(lambdas)))
        ax.set_yticklabels([f"{l:g}" for l in lambdas], fontsize=7)
        ax.set_xlabel(r"$\beta$", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(r"$\lambda$", fontsize=8)
        ax.set_title(MODEL_LABEL[model], fontsize=8.5)
        ax.axvline(x=betas.index(0.9) if 0.9 in betas else 7.5, color=C_RED, linewidth=1.0, linestyle="--")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("NDCG@10", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.text(0.5, -0.02, r"Dashed line: $\beta=0.9$ (original search-grid boundary); "
              r"extension to $\beta=2.0$ shows the plateau.", ha="center", fontsize=6.8, color=MUTED)
    fig.savefig(C.FIGURES_DIR / "fig_lambda_beta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_lambda_beta_heatmap.pdf")


def fig_ablation() -> None:
    stats_df = pd.read_csv(C.RESULTS_DIR / "stats_full.csv")
    ablation = stats_df[stats_df.family == "E_Ablation_vs_MV"]
    pivot = ablation.pivot_table(index="comparison", columns="metric", values="d_impression", aggfunc="mean")

    variants = [f"{name} vs mv_eac" for name in C.ABLATION_CONFIGS]
    variant_labels = [f"No-{name.split('_')[1].capitalize()}" for name in C.ABLATION_CONFIGS]
    metrics = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
    metric_labels = ["NDCG@10", "ILD@10", "Entropy@10", "ERR@10", "TCI@10"]

    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    y0 = np.arange(len(metrics))
    bar_height = 0.8 / len(variants)
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    for i, (variant, label) in enumerate(zip(variants, variant_labels)):
        values = [pivot.loc[variant, m] if variant in pivot.index else np.nan for m in metrics]
        y = y0 + (i - (len(variants) - 1) / 2) * bar_height
        ax.barh(y, values, height=bar_height * 0.92, color=colors[i], label=label)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y0)
    ax.set_yticklabels(metric_labels, fontsize=8)
    ax.set_xlabel("Cohen's $d$ (variant vs. Full MV-EAC)", fontsize=8)
    ax.invert_yaxis()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(C.FIGURES_DIR / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_ablation.pdf")


def fig_weight_sweep() -> None:
    summary = pd.read_csv(C.RESULTS_DIR / "test_summary.csv")
    sweep = summary[summary.method.str.startswith("sweep_wtopic_")].copy()
    sweep["w"] = sweep.method.str.extract(r"sweep_wtopic_([\d.]+)").astype(float)
    uniform = summary[summary.method == "mv_eac"].copy()
    uniform["w"] = 1 / 3
    sweep = pd.concat([sweep, uniform], ignore_index=True)

    metrics = [("NDCG@K", "NDCG@10"), ("ILD@K", "ILD@10"), ("Entropy", "Entropy@10"), ("TCI@K", "TCI@10")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), constrained_layout=True)
    for ax, (column, title) in zip(axes, metrics):
        grouped = sweep.groupby("w")[column].mean().sort_index()
        ax.plot(grouped.index, grouped.values, color=C_BLUE, linewidth=2.0, marker="o", markersize=3.5)
        ax.axvline(1 / 3, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(r"$w_{\mathrm{topic}}$", fontsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        _style(ax)
        ax.grid(axis="both")
    fig.text(0.5, -0.06, "Category and entity weights split the remainder equally "
              r"(dotted: uniform, $w=1/3$).", ha="center", fontsize=6.8, color=MUTED)
    fig.savefig(C.FIGURES_DIR / "fig_weight_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_weight_sweep.pdf")


if __name__ == "__main__":
    fig_method_comparison()
    fig_lambda_beta_heatmap()
    fig_ablation()
    fig_weight_sweep()
    print(f"\nFigures written to {C.FIGURES_DIR}")
