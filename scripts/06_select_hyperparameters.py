#!/usr/bin/env python3
"""
Step 6 of 10 -- MAUT hyperparameter selection.

Reads every ``val_grid_{model}_{method}.csv`` written by step 5 and picks one
(lambda, beta) per (model, method) via the MAUT criterion (``mveac.evaluation.maut``):
NDCG@K, Entropy, ERR@K, TCI@K at 0.25 weight each, beta capped at 0.9 with a
small parsimony tolerance (beta in {1.0,1.5,2.0} is evaluated only for the
sensitivity table written to beta_sensitivity.csv). Trad-Cal's lambda is selected independently from
the beta=0 slice of the same grid.

Usage:
    python scripts/06_select_hyperparameters.py

Output (under $MVEAC_DATA_ROOT/results/):
    best_params.csv   one row per (model, method): eac_lambda, eac_beta, trad_lambda
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.evaluation.maut import maut_score, select_best_eac, select_best_traditional, select_unrestricted

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    rows = []
    sens_rows = []
    missing = []
    for model in C.MODELS:
        for method in C.GRID_METHODS:
            path = C.RESULTS_DIR / f"val_grid_{model}_{method}.csv"
            if not path.exists():
                missing.append(path.name)
                continue
            expected = len(C.LAMBDA_GRID) * len(C.BETA_GRID)
            grid = pd.read_csv(path)
            if len(grid) < expected:
                missing.append(f"{path.name} ({len(grid)}/{expected} rows)")
                continue

            grid["maut_score"] = maut_score(grid)
            eac_best = select_best_eac(grid)
            trad_best = select_best_traditional(grid)
            unrestricted = select_unrestricted(grid)
            if (unrestricted["lambda"], unrestricted["beta"]) != (eac_best["lambda"], eac_best["beta"]):
                sens_rows.append({
                    "model": model, "method": method,
                    "reported_lambda": float(eac_best["lambda"]), "reported_beta": float(eac_best["beta"]),
                    "unrestricted_lambda": float(unrestricted["lambda"]),
                    "unrestricted_beta": float(unrestricted["beta"]),
                    **{f"delta_{m}": float(unrestricted[m] - eac_best[m]) for m in C.MAUT_METRICS},
                })

            rows.append({
                "model": model, "method": method,
                "eac_lambda": float(eac_best["lambda"]), "eac_beta": float(eac_best["beta"]),
                "trad_lambda": float(trad_best["lambda"]),
                "NDCG@K": float(eac_best["NDCG@K"]), "Entropy": float(eac_best["Entropy"]),
                "ERR@K": float(eac_best["ERR@K"]), "TCI@K": float(eac_best["TCI@K"]),
            })

    if missing:
        log.error("Grid search incomplete -- rerun scripts/05_grid_search.py:\n  %s", "\n  ".join(missing))
        return

    best = pd.DataFrame(rows)
    best.to_csv(C.RESULTS_DIR / "best_params.csv", index=False)
    log.info("Selected hyperparameters:\n%s", best.to_string(index=False))
    pd.DataFrame(sens_rows).to_csv(C.RESULTS_DIR / "beta_sensitivity.csv", index=False)
    log.info("%d/%d EAC selections would change if beta > 0.9 were eligible "
             "(paper Table 'beta sensitivity'; written to beta_sensitivity.csv)", len(sens_rows), len(best))


if __name__ == "__main__":
    main()
