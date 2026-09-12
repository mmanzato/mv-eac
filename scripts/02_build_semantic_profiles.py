#!/usr/bin/env python3
"""
Step 2 of 10 -- build the five semantic-view vocabularies, article maps, and
per-user preference profiles that every calibration/EAC method reranks
against.

User profiles are built only for the users appearing in the val/test
subsamples (typically ~250-300K distinct users out of ~790K in the full
dataset) -- building all five views for every user in EB-NeRD Large would use
far more memory than the entity view alone needs for the subsample users
(~5,852-dim float32 vector x ~290K users ~= 7 GB).

Usage:
    python scripts/02_build_semantic_profiles.py

Output (under $MVEAC_DATA_ROOT/processed/semantic_spaces/):
    vocabs.json            {view: [values]}
    article_maps.pkl       {view: {article_id: [values]}}
    category_vectors.pkl   {article_id: one-hot category vector} (for ILD@K)
    user_profiles.pkl       {view: {user_id: profile vector}}
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.data.loader import load_articles, load_train_behaviors, load_train_history, build_user_history_map
from mveac.data.semantic_profiles import (
    build_all_vocabs,
    build_article_maps,
    build_category_vectors,
    build_entity_vocab,
    build_user_profiles,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    out_vocabs = C.SEM_DIR / "vocabs.json"
    out_maps = C.SEM_DIR / "article_maps.pkl"
    out_profiles = C.SEM_DIR / "user_profiles.pkl"
    if out_vocabs.exists() and out_maps.exists() and out_profiles.exists():
        log.info("Semantic spaces already built at %s -- skipping.", C.SEM_DIR)
        return

    log.info("Loading article metadata ...")
    articles = load_articles(C.EBNERD_LARGE_ZIP)

    log.info("Building vocabularies ...")
    entity_vocab_list = build_entity_vocab(articles, C.ENTITY_MIN_FREQ)
    vocabs = build_all_vocabs(articles, C.ENTITY_MIN_FREQ, C.SENTIMENT_LABELS)
    for view, vocab in vocabs.items():
        log.info("  %-12s %d values", view, len(vocab))

    log.info("Building article maps ...")
    article_maps = build_article_maps(articles, entity_vocab=set(entity_vocab_list))

    log.info("Building one-hot category vectors (for ILD@K) ...")
    category_vectors = build_category_vectors(articles, article_maps["category"], vocabs["category"])

    log.info("Identifying subsample users ...")
    val_sub = pd.read_parquet(C.PROC_DIR / "val_sub.parquet")
    test_sub = pd.read_parquet(C.PROC_DIR / "test_sub.parquet")
    target_users = set(val_sub["user_id"].astype(int)) | set(test_sub["user_id"].astype(int))
    log.info("Building profiles for %d unique users ...", len(target_users))

    log.info("Loading train history + behaviors to build reading histories ...")
    history_df = load_train_history(C.EBNERD_LARGE_ZIP)
    train_behaviors = load_train_behaviors(C.EBNERD_LARGE_ZIP)
    user_history = build_user_history_map(history_df, train_behaviors, target_users=target_users)
    del history_df, train_behaviors

    log.info("Building per-view user profiles (float32) ...")
    user_profiles: dict[str, dict[int, np.ndarray]] = {}
    for view in C.ALL_VIEWS:
        profiles = build_user_profiles(user_history, article_maps[view], vocabs[view])
        user_profiles[view] = {uid: arr.astype(np.float32) for uid, arr in profiles.items()}
        approx_gb = len(user_profiles[view]) * len(vocabs[view]) * 4 / 1e9
        log.info("  %-12s %d profiles (~%.2f GB)", view, len(user_profiles[view]), approx_gb)

    C.SEM_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_vocabs, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False)
    with open(out_maps, "wb") as f:
        pickle.dump(article_maps, f, protocol=4)
    with open(C.SEM_DIR / "category_vectors.pkl", "wb") as f:
        pickle.dump(category_vectors, f, protocol=4)
    with open(out_profiles, "wb") as f:
        pickle.dump(user_profiles, f, protocol=4)
    log.info("Semantic spaces saved to %s", C.SEM_DIR)


if __name__ == "__main__":
    main()
