#!/usr/bin/env python3
"""
Step 4 of 10 -- train NRMS and score the val/test subsamples.

The primary run (``--tag seed0``, the default) is what every other pipeline
step reranks by default. The paper additionally reports a reproducibility
check across three further independently trained instances -- two more
random seeds at the same 200,000-impression training sample, and one seed at
a five-times larger 1,000,000-impression sample -- to verify the results are
not an artifact of a single, possibly under-trained, model instance. Each
variant is completely independent (its own random seed, its own training
subsample of the ~12M training behaviors, its own checkpoint) but scores the
same val/test impressions so the outputs are directly comparable.

Usage:
    python scripts/04_train_nrms.py                          # primary seed0 run
    python scripts/04_train_nrms.py --tag seed1 --seed 1
    python scripts/04_train_nrms.py --tag seed2 --seed 2
    python scripts/04_train_nrms.py --tag large1M --seed 0 --train-sample 1000000

Output (under $MVEAC_DATA_ROOT/):
    processed/scores/scores_nrms_{tag}_val.parquet / _test.parquet
    checkpoints/nrms_{tag}.pt
    logs/nrms_{tag}.log
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.data.loader import (
    build_embedding_matrix,
    build_user_history_map,
    load_articles,
    load_train_behaviors,
    load_train_history,
    load_word2vec_embeddings,
)
from mveac.models.nrms import NRMS


def _save_scores(scores: dict[int, list[tuple[int, float]]], tag: str, split: str) -> None:
    rows = [
        {"impression_id": imp_id, "article_id": aid, "rank": rank, "score": score}
        for imp_id, ranked in scores.items()
        for rank, (aid, score) in enumerate(ranked, start=1)
    ]
    path = C.SCORES_DIR / f"scores_nrms_{tag}_{split}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    logging.info("Saved %s (%d impressions)", path.name, len(scores))


def run(tag: str, seed: int, train_sample: int) -> None:
    log_path = C.LOGS_DIR / f"nrms_{tag}.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger(tag)
    log.info("=== NRMS training: tag=%s seed=%d train_sample=%d ===", tag, seed, train_sample)

    val_path = C.SCORES_DIR / f"scores_nrms_{tag}_val.parquet"
    test_path = C.SCORES_DIR / f"scores_nrms_{tag}_test.parquet"
    if val_path.exists() and test_path.exists():
        log.info("Scores already exist for tag=%s -- skipping.", tag)
        return

    random.seed(seed)
    np.random.seed(seed)

    log.info("Loading Word2Vec embeddings ...")
    w2v_df = load_word2vec_embeddings(C.WORD2VEC_ZIP)
    articles = load_articles(C.EBNERD_LARGE_ZIP)
    emb_matrix, id2idx = build_embedding_matrix(articles["article_id"].tolist(), w2v_df)
    log.info("Embedding matrix: %d x %d", *emb_matrix.shape)
    del articles, w2v_df

    log.info("Loading training behaviors, sampling %d (seed=%d) ...", train_sample, seed)
    train_behaviors = load_train_behaviors(C.EBNERD_LARGE_ZIP)
    train_behaviors = train_behaviors[train_behaviors["article_ids_clicked"].apply(len) > 0]
    if len(train_behaviors) > train_sample:
        train_behaviors = train_behaviors.sample(n=train_sample, random_state=seed)
    log.info("Training behaviors: %d impressions, %d users",
              len(train_behaviors), train_behaviors["user_id"].nunique())

    train_users = set(train_behaviors["user_id"].astype(int))
    history_df = load_train_history(C.EBNERD_LARGE_ZIP)
    empty_behaviors = pd.DataFrame(columns=["user_id", "article_ids_clicked", "impression_time"])
    train_history = build_user_history_map(history_df, empty_behaviors, target_users=train_users)
    del history_df

    model = NRMS(
        input_dim=emb_matrix.shape[1], hidden_dim=C.NRMS_HIDDEN_DIM, num_heads=C.NRMS_NUM_HEADS,
        dropout=C.NRMS_DROPOUT, epochs=C.NRMS_EPOCHS, batch_size=C.NRMS_BATCH_SIZE, lr=C.NRMS_LR,
        neg_ratio=C.NRMS_NEG_RATIO, max_history=C.NRMS_MAX_HISTORY, seed=seed,
        checkpoint_path=C.CKPT_DIR / f"nrms_{tag}.pt",
    )
    log.info("Training on device=%s ...", model.device)
    model.fit(train_behaviors, train_history, emb_matrix, id2idx)
    del train_behaviors, train_history

    log.info("Building reading history for val/test users ...")
    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    eval_users = set(val_sub["user_id"].astype(int)) | set(test_sub["user_id"].astype(int))
    history_df = load_train_history(C.EBNERD_LARGE_ZIP)
    train_behaviors_full = load_train_behaviors(C.EBNERD_LARGE_ZIP)
    eval_history = build_user_history_map(history_df, train_behaviors_full, target_users=eval_users)
    del history_df, train_behaviors_full

    if not val_path.exists():
        log.info("Scoring val (%d impressions) ...", len(val_sub))
        _save_scores(model.score_all_impressions(val_sub, eval_history), tag, "val")
    if not test_path.exists():
        log.info("Scoring test (%d impressions) ...", len(test_sub))
        _save_scores(model.score_all_impressions(test_sub, eval_history), tag, "test")

    log.info("=== tag=%s complete ===", tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="seed0", help="Identifies this training run's scores/checkpoint.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-sample", type=int, default=C.NRMS_TRAIN_SAMPLE)
    args = ap.parse_args()
    run(args.tag, args.seed, args.train_sample)
