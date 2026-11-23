#!/usr/bin/env python3
"""
Second revision, step 33 -- the result figures of the revised paper, from the step-21/23/19-20 outputs,
in the style of step 10.

  fig_method_comparison.pdf    method - Original for Trad-Cal-MV, MV-EAC, Topic-EAC, Entity-EAC, xQuAD (Figure 3)
  fig_lambda_beta_heatmap.pdf  MV-EAC validation NDCG@10 over the lambda x beta grid; selected cell outlined (Figure 2)
  fig_ablation.pdf             pooled d of the ablations and entity controls vs MV-EAC (Figure 4)
  fig_weight_sweep.pdf         the topic-weight sweep (Figure 5)

Output: $MVEAC_DATA_ROOT/figures/rev2/
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C


class MF:   # palette and style of step 10
    C_BLUE, C_ORANGE, C_AQUA, C_RED, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7"
    MODEL_COLOR = {"most_popular": C_BLUE, "itemknn": C_ORANGE, "nrms": C_AQUA}
    MODEL_LABEL = {"most_popular": "Pop", "itemknn": "KNN", "nrms": "NRMS"}
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

    @staticmethod
    def _style_ax(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x")


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MF.GRID, "axes.labelcolor": MF.INK, "text.color": MF.INK,
    "xtick.color": MF.MUTED, "ytick.color": MF.MUTED,
    "axes.grid": True, "grid.color": MF.GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "svg.fonttype": "none",
})

REV2 = C.RESULTS_DIR / "rev2"
FIGS = C.FIGURES_DIR / "rev2"
FIGS.mkdir(parents=True, exist_ok=True)
MODELS = list(C.MODELS)


def fig_method_comparison():
    ts = pd.read_csv(REV2 / "test_summary_rev2.csv")
    methods = ["tradcal_mv_norm", "mv_eac_norm", "topic_eac", "entity_eac", "xquad"]
    labels = ["Trad-Cal-MV", "MV-EAC", "Topic-EAC", "Entity-EAC", "xQuAD"]
    metrics = [("NDCG@K", "NDCG@10", True), ("ILD@K", "ILD@10", True), ("Entropy", "Entropy@10", True),
               ("ERR@K", "ERR@10", False), ("TCI@K", "TCI@10", False)]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.4), constrained_layout=True)
    bw = 0.8 / len(MODELS)
    for ax, (col, title, hb) in zip(axes, metrics):
        for mi, model in enumerate(MODELS):
            g = ts[ts.model == model].set_index("config")[col]
            vals = [g[m] - g["original"] for m in methods]
            x = np.arange(len(methods)) + (mi - (len(MODELS) - 1) / 2) * bw
            ax.bar(x, vals, width=bw * 0.92, color=MF.MODEL_COLOR[model], label=MF.MODEL_LABEL[model])
        ax.axhline(0, color=MF.INK, linewidth=0.7)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.6)
        ax.set_title(f"Δ{title} {'↑' if hb else '↓'}", fontsize=8.2)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4)); ax.tick_params(axis="y", labelsize=6.5)
        MF._style_ax(ax)
    axes[0].set_ylabel("method − Original", fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=8)
    fig.savefig(FIGS / "fig_method_comparison.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote fig_method_comparison.pdf")


def fig_lambda_beta_heatmap():
    bp = pd.read_csv(REV2 / "best_params_rev2.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        g = pd.read_csv(REV2 / f"val_grid_{model}_mv_eac_norm.csv")
        lams, bets = sorted(g["lambda"].unique()), sorted(g["beta"].unique())
        mat = np.full((len(lams), len(bets)), np.nan)
        for _, r in g.iterrows():
            mat[lams.index(r["lambda"]), bets.index(r["beta"])] = r["NDCG@K"]
        im = ax.imshow(mat, aspect="auto", cmap="Blues", origin="lower")
        ax.set_xticks(range(len(bets))); ax.set_xticklabels([f"{b:g}" for b in bets], rotation=60, fontsize=6.0)
        ax.set_yticks(range(len(lams))); ax.set_yticklabels([f"{x:g}" for x in lams], fontsize=7)
        ax.set_xlabel(r"$\beta$", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(r"$\lambda$", fontsize=8)
        ax.set_title(MF.MODEL_LABEL[model], fontsize=8.5)
        s = bp[(bp.model == model) & (bp.config == "mv_eac_norm")].iloc[0]
        i, j = lams.index(s["lambda"]), bets.index(s["beta"])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=MF.C_RED, linewidth=1.3))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02); cb.ax.tick_params(labelsize=6)
        if ax is axes[-1]:
            cb.set_label("NDCG@10", fontsize=7.5)
    fig.savefig(FIGS / "fig_lambda_beta_heatmap.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote fig_lambda_beta_heatmap.pdf")


def fig_ablation():
    """Pooled Cohen's d (raw sign) of each view variant vs MV-EAC: the three ablations and the two
    entity controls (No-Entity with fixed 1/3 weights; Cat+Topic with its own selection)."""
    s = pd.read_csv(REV2 / "stats" / "pooled_rev2.csv")
    rows = [("E_Ablation_vs_MV", "no_category vs mv_eac_norm", "No-Category", MF.C_BLUE, None),
            ("E_Ablation_vs_MV", "no_entity vs mv_eac_norm", "No-Entity", MF.C_ORANGE, None),
            ("E_Ablation_vs_MV", "no_topic vs mv_eac_norm", "No-Topic", MF.C_AQUA, None),
            ("N_D1_EntityControls", "no_entity_fixed vs mv_eac_norm", "No-Entity (fixed)", MF.C_ORANGE, "////"),
            ("N_D1_EntityControls", "cat_topic_tuned vs mv_eac_norm", "Cat+Topic (tuned)", MF.C_VIOLET, None)]
    metrics = ["NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"]
    mlab = ["NDCG@10", "ILD@10", "Entropy@10", "ERR@10", "TCI@10"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    n = len(rows); bh = 0.84 / n; y0 = np.arange(len(metrics))
    for k, (fam, comp, lab, col, hatch) in enumerate(rows):
        x = s[(s.family == fam) & (s.comparison == comp)].set_index("metric")["d_pooled"]
        y = y0 + (k - (n - 1) / 2) * bh
        ax.barh(y, [x[m] for m in metrics], height=bh * 0.92, color=col, label=lab,
                hatch=hatch, edgecolor="white" if hatch else None, linewidth=0)
    ax.axvline(0, color=MF.INK, linewidth=0.8)
    ax.set_yticks(y0); ax.set_yticklabels(mlab, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("pooled Cohen's $d$ of (variant $-$ MV-EAC)\n"
                  "ERR@10, TCI@10: lower is better, so $d>0$ is a degradation", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.savefig(FIGS / "fig_ablation.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote fig_ablation.pdf")


def fig_weight_sweep():
    ts = pd.read_csv(REV2 / "test_summary_rev2.csv")
    sw = ts[ts.config.str.startswith("sweep_wtopic_")].copy()
    sw["w"] = sw.config.str.extract(r"sweep_wtopic_([\d.]+)", expand=False).astype(float)
    ends = [(ts[ts.config == "mv_eac_norm"], 1 / 3), (ts[ts.config == "ablation_no_topic"], 0.0)]
    sw = pd.concat([sw] + [e.assign(w=w) for e, w in ends], ignore_index=True)
    metrics = [("NDCG@K", "NDCG@10"), ("ILD@K", "ILD@10"), ("Entropy", "Entropy@10"), ("ERR@K", "ERR@10"), ("TCI@K", "TCI@10")]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.9), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        g = sw.groupby("w")[col].mean().sort_index()
        ax.plot(g.index, g.values, color=MF.C_BLUE, linewidth=2.0, marker="o", markersize=3.5)
        ax.axvline(1 / 3, color=MF.MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=8.5); ax.set_xlabel(r"$w_{\mathrm{topic}}$", fontsize=8)
        ax.set_xticks([0, 0.15, 1 / 3, 0.5, 0.7, 0.85, 1])
        ax.set_xticklabels(["0", ".15", "1/3", ".5", ".7", ".85", "1"], fontsize=5.5, rotation=90)
        ax.tick_params(axis="y", labelsize=6.5); ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        MF._style_ax(ax); ax.grid(axis="both")
    fig.text(0.5, -0.06, "Category and entity weights split the remainder equally (dotted: uniform, $w=1/3$).",
             ha="center", fontsize=6.8, color=MF.MUTED)
    fig.savefig(FIGS / "fig_weight_sweep.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote fig_weight_sweep.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["method", "heatmap", "ablation", "sweep"]
    if "method" in which:
        fig_method_comparison()
    if "heatmap" in which:
        fig_lambda_beta_heatmap()
    if "ablation" in which:
        fig_ablation()
    if "sweep" in which:
        fig_weight_sweep()
