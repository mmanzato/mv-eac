"""
Readers for the raw EB-NeRD Large dataset.

The dataset ships as two zip archives (see ``docs/data.md`` for download
instructions):

    ebnerd_large.zip
        articles.parquet                  article metadata (category, topics,
                                           named entities, sentiment, ...)
        train/behaviors.parquet           ~12.06M training impressions
        train/history.parquet             per-user click history before the
                                           training window
        validation/behaviors.parquet      ~12.57M validation impressions (this
                                           project draws ALL of its evaluation
                                           subsamples from here -- EB-NeRD does
                                           not ship a separate held-out test
                                           split, see docs/pipeline.md)

    Ekstra_Bladet_word2vec.zip
        Ekstra_Bladet_word2vec/document_vector.parquet
                                           pre-trained 300-d Word2Vec document
                                           embedding per article, used as the
                                           frozen article representation for
                                           both ItemKNN and NRMS.

We read parquet members directly out of the zip archives (via an in-memory
BytesIO buffer) rather than extracting them to disk first, since the archives
together are only marginally smaller than the disk space two full extracted
copies would need.
"""
from __future__ import annotations

import io
import logging
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _read_parquet_member(zip_path: Path, member: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a single parquet file out of a zip archive without extracting it."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. See docs/data.md for download instructions."
        )
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            return pd.read_parquet(io.BytesIO(fh.read()), columns=columns)


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def load_articles(ebnerd_zip: Path) -> pd.DataFrame:
    """Load and lightly normalize the article metadata table.

    Casts ``article_id`` to int, fills missing categories with "unknown", and
    converts the ``subcategory`` array column to plain Python lists so it can
    be used as a dict value elsewhere in the pipeline.
    """
    df = _read_parquet_member(ebnerd_zip, "articles.parquet")
    df["article_id"] = df["article_id"].astype(int)
    df["category_str"] = df["category_str"].fillna("unknown").astype(str)
    df["subcategory"] = df["subcategory"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else []
    )
    log.info("Articles loaded: %d | categories: %d", len(df), df["category_str"].nunique())
    return df


# ---------------------------------------------------------------------------
# Behaviors (impressions)
# ---------------------------------------------------------------------------

def _parse_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(int)
    df["impression_id"] = df["impression_id"].astype(int)
    df["impression_time"] = pd.to_datetime(df["impression_time"], utc=False)
    df["article_ids_inview"] = df["article_ids_inview"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )
    df["article_ids_clicked"] = df["article_ids_clicked"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )
    return df


def load_train_behaviors(ebnerd_zip: Path) -> pd.DataFrame:
    """Load the ~12.06M-impression training behavior log."""
    df = _parse_behaviors(_read_parquet_member(ebnerd_zip, "train/behaviors.parquet"))
    log.info("Train behaviors: %d impressions | %d users", len(df), df["user_id"].nunique())
    return df


def load_validation_behaviors(ebnerd_zip: Path) -> pd.DataFrame:
    """Load the ~12.57M-impression validation behavior log.

    This is the *pool* from which this project's own 50K/500K evaluation
    subsamples are drawn (Section 4.1 of the paper) -- EB-NeRD Large ships no
    separate held-out test partition.
    """
    df = _parse_behaviors(_read_parquet_member(ebnerd_zip, "validation/behaviors.parquet"))
    log.info("Validation behaviors (pool): %d impressions | %d users", len(df), df["user_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# User history
# ---------------------------------------------------------------------------

def load_train_history(ebnerd_zip: Path) -> pd.DataFrame:
    """Load each user's click history recorded *before* the training window."""
    df = _read_parquet_member(ebnerd_zip, "train/history.parquet")
    df["user_id"] = df["user_id"].astype(int)
    df["article_id_fixed"] = df["article_id_fixed"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )
    log.info("Train history: %d users", len(df))
    return df


def build_user_history_map(
    history_df: pd.DataFrame,
    train_behaviors: pd.DataFrame,
    target_users: set[int] | None = None,
) -> dict[int, list[int]]:
    """Build a ``user_id -> [article_id, ...]`` reading-history map.

    Combines the pre-training history log with clicks the user made during
    the training window itself (in chronological order), since both are valid
    signal for building that user's semantic profile. Pass ``target_users`` to
    restrict construction to a specific set of users -- building the map for
    all ~790K users in EB-NeRD Large at once is unnecessary (and memory-heavy)
    when only ~20-300K evaluation-subsample users are needed downstream.
    """
    pre_hist: dict[int, list[int]] = {}
    for row in history_df.itertuples(index=False):
        uid = int(row.user_id)
        if target_users is not None and uid not in target_users:
            continue
        pre_hist[uid] = [int(x) for x in row.article_id_fixed]

    train_clicks: dict[int, list[int]] = {}
    if "article_ids_clicked" in train_behaviors.columns and len(train_behaviors):
        sub = train_behaviors.sort_values("impression_time")[["user_id", "article_ids_clicked"]]
        for row in sub.itertuples(index=False):
            uid = int(row.user_id)
            if target_users is not None and uid not in target_users:
                continue
            clicked = [int(x) for x in row.article_ids_clicked]
            if clicked:
                train_clicks.setdefault(uid, []).extend(clicked)

    all_users = set(pre_hist) | set(train_clicks)
    combined = {uid: pre_hist.get(uid, []) + train_clicks.get(uid, []) for uid in all_users}
    log.info("User history built for %d users", len(combined))
    return combined


def compute_click_counts(train_behaviors: pd.DataFrame) -> dict[int, int]:
    """Total click count per article in the training behaviors -- the Most Popular score."""
    counts: dict[int, int] = {}
    for clicks in train_behaviors["article_ids_clicked"]:
        for aid in clicks:
            counts[int(aid)] = counts.get(int(aid), 0) + 1
    log.info("Click counts computed: %d distinct clicked articles", len(counts))
    return counts


# ---------------------------------------------------------------------------
# Word2Vec article embeddings
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_word2vec_embeddings(word2vec_zip: Path) -> pd.DataFrame:
    """Load the pre-trained 300-d Word2Vec document vectors, indexed by article_id."""
    df = _read_parquet_member(word2vec_zip, "Ekstra_Bladet_word2vec/document_vector.parquet")
    df["article_id"] = df["article_id"].astype(int)
    df = df.set_index("article_id")
    dim = len(df["document_vector"].iloc[0]) if len(df) else 0
    log.info("Word2Vec embeddings loaded: %d articles | dim=%d", len(df), dim)
    return df


def build_embedding_matrix(
    article_ids: list[int],
    embeddings_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, int]]:
    """Stack per-article Word2Vec vectors into a dense matrix.

    Returns ``(matrix, id2idx)`` where ``matrix[id2idx[aid]]`` is the embedding
    for article ``aid``. Articles absent from ``embeddings_df`` are dropped.
    """
    valid_ids = [aid for aid in article_ids if aid in embeddings_df.index]
    if not valid_ids:
        return np.zeros((0, 300), dtype=np.float32), {}
    vecs = [np.asarray(embeddings_df.loc[aid, "document_vector"], dtype=np.float32) for aid in valid_ids]
    matrix = np.stack(vecs, axis=0)
    id2idx = {aid: i for i, aid in enumerate(valid_ids)}
    return matrix, id2idx
