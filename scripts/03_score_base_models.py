#!/usr/bin/env python3
"""
Step 3 of 10 -- score the val/test subsamples with the two non-neural base
recommenders (Most Popular, ItemKNN-W2V). NRMS is trained and scored
separately (step 4), since it needs a GPU/MPS-capable training loop rather
than a closed-form fit.

Usage:
    python scripts/03_score_base_models.py

Output (under $MVEAC_DATA_ROOT/processed/scores/):
    scores_most_popular_val.parquet / _test.parquet
    scores_itemknn_val.parquet / _test.parquet
Each parquet has columns [impression_id, article_id, rank, score], one row
per (impression, candidate) pair.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.data.loader import (
    build_embedding_matrix,
    build_user_history_map,
    compute_click_counts,
    load_articles,
    load_train_behaviors,
    load_train_history,
    load_word2vec_embeddings,
)
from mveac.models.item_knn import ItemKNN
from mveac.models.most_popular import MostPopular

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _save_scores(scores: dict[int, list[tuple[int, float]]], model: str, split: str) -> None:
    rows = [
        {"impression_id": imp_id, "article_id": aid, "rank": rank, "score": score}
        for imp_id, ranked in scores.items()
        for rank, (aid, score) in enumerate(ranked, start=1)
    ]
    path = C.SCORES_DIR / f"scores_{model}_{split}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    log.info("Saved %s (%d impressions)", path.name, len(scores))


def main() -> None:
    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    log.info("val=%d impressions, test=%d impressions", len(val_sub), len(test_sub))

    # ---- Most Popular ------------------------------------------------------
    mp_paths = [C.SCORES_DIR / f"scores_most_popular_{s}.parquet" for s in ("val", "test")]
    if not all(p.exists() for p in mp_paths):
        log.info("--- Most Popular ---")
        train_behaviors = load_train_behaviors(C.EBNERD_LARGE_ZIP)
        click_counts = compute_click_counts(train_behaviors)
        del train_behaviors
        model = MostPopular().fit(click_counts)
        if not mp_paths[0].exists():
            _save_scores(model.score_all_impressions(val_sub), "most_popular", "val")
        if not mp_paths[1].exists():
            _save_scores(model.score_all_impressions(test_sub), "most_popular", "test")
    else:
        log.info("Most Popular scores already exist -- skipping.")

    # ---- ItemKNN-W2V ---------------------------------------------------------
    knn_paths = [C.SCORES_DIR / f"scores_itemknn_{s}.parquet" for s in ("val", "test")]
    if not all(p.exists() for p in knn_paths):
        log.info("--- ItemKNN-W2V ---")
        w2v_df = load_word2vec_embeddings(C.WORD2VEC_ZIP)
        articles = load_articles(C.EBNERD_LARGE_ZIP)
        emb_matrix, id2idx = build_embedding_matrix(articles["article_id"].tolist(), w2v_df)
        log.info("Embedding matrix: %d articles x %d dims", *emb_matrix.shape)
        del articles, w2v_df

        model = ItemKNN().fit(emb_matrix, id2idx)

        target_users = set(val_sub["user_id"].astype(int)) | set(test_sub["user_id"].astype(int))
        history_df = load_train_history(C.EBNERD_LARGE_ZIP)
        train_behaviors = load_train_behaviors(C.EBNERD_LARGE_ZIP)
        user_history = build_user_history_map(history_df, train_behaviors, target_users=target_users)
        del history_df, train_behaviors

        if not knn_paths[0].exists():
            _save_scores(model.score_all_impressions(val_sub, user_history), "itemknn", "val")
        if not knn_paths[1].exists():
            _save_scores(model.score_all_impressions(test_sub, user_history), "itemknn", "test")
    else:
        log.info("ItemKNN scores already exist -- skipping.")

    log.info("Base-model scoring complete -> %s", C.SCORES_DIR)


if __name__ == "__main__":
    main()
