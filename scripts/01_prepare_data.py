#!/usr/bin/env python3
"""
Step 1 of 10 -- draw the stratified validation/test subsamples used everywhere
else in the pipeline.

Reads the raw EB-NeRD Large validation behaviors (~12.57M impressions) and
draws two disjoint stratified subsamples from it: 50,000 impressions for
hyperparameter selection and 500,000 for the final reported evaluation (see
``mveac.data.sampling`` for the stratification scheme). Also prints the
candidate-set-size report used in the paper's Table 1.

Usage:
    python scripts/01_prepare_data.py

Output (under $MVEAC_DATA_ROOT/processed/, default ./data/processed/):
    val_sub.parquet, test_sub.parquet
    dataset_report.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.data.loader import load_train_history, load_validation_behaviors, build_user_history_map
from mveac.data.sampling import build_val_test_subsamples, candidate_set_size_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    val_path, test_path = C.PROC_DIR / "val_sub.parquet", C.PROC_DIR / "test_sub.parquet"
    if val_path.exists() and test_path.exists():
        log.info("Subsamples already exist at %s -- skipping.", C.PROC_DIR)
        return

    log.info("Loading EB-NeRD Large validation behaviors (this is the ~12.57M-impression pool) ...")
    val_behaviors = load_validation_behaviors(C.EBNERD_LARGE_ZIP)

    log.info("Loading training-window history for stratification by history length ...")
    history_df = load_train_history(C.EBNERD_LARGE_ZIP)
    target_users = set(val_behaviors["user_id"].astype(int))
    # Only the pre-training history log is needed for a history-length quartile
    # (not the training clicks build_user_history_map can additionally fold in),
    # so pass an empty behaviors frame of the right shape.
    empty_behaviors = pd.DataFrame(columns=["user_id", "article_ids_clicked", "impression_time"])
    history_map = build_user_history_map(history_df, empty_behaviors, target_users=target_users)
    history_sizes = {uid: len(arts) for uid, arts in history_map.items()}
    del history_df, history_map

    log.info("Drawing stratified subsamples (val=%d, test=%d) ...", C.VAL_SUBSAMPLE, C.TEST_SUBSAMPLE)
    val_sub, test_sub = build_val_test_subsamples(
        val_behaviors, history_sizes,
        val_n=C.VAL_SUBSAMPLE, test_n=C.TEST_SUBSAMPLE, n_quantiles=C.N_QUANTILES,
        val_seed=C.VAL_SAMPLE_SEED, test_seed=C.TEST_SAMPLE_SEED,
    )
    val_sub.to_parquet(val_path, index=False)
    test_sub.to_parquet(test_path, index=False)
    log.info("Saved %s and %s", val_path, test_path)

    report = {
        "val": candidate_set_size_report(val_sub, C.K),
        "test": candidate_set_size_report(test_sub, C.K),
    }
    with open(C.PROC_DIR / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Candidate-set-size report (paper Table 1):\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
