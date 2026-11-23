#!/usr/bin/env python3
"""
Second revision, step 20 -- MAUT selections from the step-19 grids.

Same equal-weight four-metric MAUT as the paper, but every beta is eligible (no BETA_CAP);
configurations within PARSIMONY_TOL (0.003) of the best are ties, broken by the smallest beta,
then the smallest lambda. Trad-Cal maximizes the same MAUT (normalized over the paired grid with
beta <= 10) over its beta = 0 slice.

  --stage a   EAC methods (Entity-EAC over its grid up to beta = inf, the others up to 10),
              their Trad-Cal counterparts, MV-EAC (normalized) and Trad-Cal-MV, MV-EAC_nom and
              Trad-Cal-MV_nom, and the four baselines            -> rev2/best_params_rev2.csv
  --stage b   the tuned category-topic model and the exploration-component variants; a
              component's beta is chosen by MAUT normalized over the union {beta = 0, full bonus,
              hist-only, list-only} at the method's lambda      -> rev2/best_params_rev2b.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac import revision2 as R
from mveac.evaluation.maut import maut_score

OUT = C.RESULTS_DIR / "rev2"
KEEP = ["lambda", "beta", "NDCG@K", "Entropy", "ERR@K", "TCI@K"]
TRADCAL_NAME = {"category_eac": "tradcal_category", "entity_eac": "tradcal_entity", "topic_eac": "tradcal_topic",
                "sentiment_eac": "tradcal_sentiment", "mv_eac": "tradcal_mv", "mv_eac_norm": "tradcal_mv_norm"}


def grid(model, name):
    return pd.read_csv(OUT / f"val_grid_{model}_{name}.csv")


def stage_a():
    rows = []
    for m in C.MODELS:
        for meth in C.GRID_METHODS + ["mv_eac_norm"]:
            g = grid(m, meth)
            g10 = g[g.beta <= 10.0].reset_index(drop=True)
            r = R.select(g if meth == "entity_eac" else g10)
            rows.append({"model": m, "config": meth, "n_grid": len(g if meth == "entity_eac" else g10), **r[KEEP].to_dict()})
            t = R.select_tradcal(g10)
            rows.append({"model": m, "config": TRADCAL_NAME[meth], "n_grid": len(g10), **t[KEEP].to_dict()})
        for b in C.BASELINE_GRIDS:
            r = R.select(grid(m, b))
            rows.append({"model": m, "config": b, "n_grid": len(C.BASELINE_GRIDS[b]), **r[KEEP].to_dict()})
    return pd.DataFrame(rows), "best_params_rev2.csv"


def stage_b():
    sel = pd.read_csv(OUT / "best_params_rev2.csv")
    rows = []
    for m in C.MODELS:
        r = R.select(grid(m, "norm_cat_topic_grid"))
        rows.append({"model": m, "config": "cat_topic_tuned", **r[KEEP].to_dict()})
        for meth in C.COMPONENT_METHODS:
            lam = float(sel[(sel.model == m) & (sel.config == meth)]["lambda"].iloc[0])
            f = grid(m, meth)
            f = f[((f["lambda"] - lam).abs() < 1e-9) & np.isfinite(f.beta)]
            parts = [f.assign(mode=np.where(f.beta == 0, "beta0", "full"))]
            parts += [grid(m, f"c1_{meth}_{mo}").assign(mode=mo) for mo in C.UCB_COMPONENTS]
            u = pd.concat(parts, ignore_index=True)
            u["maut_score"] = maut_score(u)
            for mo in C.UCB_COMPONENTS:
                gg = u[u["mode"] == mo]
                b = gg[gg.maut_score >= gg.maut_score.max() - C.PARSIMONY_TOL].sort_values("beta").iloc[0]
                rows.append({"model": m, "config": f"c1_{meth}_{mo}", **b[KEEP].to_dict()})
    return pd.DataFrame(rows), "best_params_rev2b.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["a", "b"], required=True)
    a = ap.parse_args()
    df, name = stage_a() if a.stage == "a" else stage_b()
    df.to_csv(OUT / name, index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
