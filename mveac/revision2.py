"""
Shared machinery of the second-revision pipeline (scripts 19-33, release 1.4.0).

A *configuration* is everything needed to rerank one split for one base model:

    Spec(name, kind, method, lam, beta, weights, normalize, ucb_mode, epsilon)

* ``kind="original"``  -- the base ranking cut to K;
* ``kind="eac"``       -- :func:`mveac.calibration.eac.rerank_all_general` over ``weights``
                          (single-view methods use weight 1 on their view, Trad-Cal is beta=0);
* ``kind="baseline"``  -- MMR / xQuAD / DPP / Dedup with trade-off parameter ``lam``.

:func:`load_state` loads the semantic spaces, user profiles, behaviors and base scores of one
split once per process (workers are forked and share it); :func:`rerank` runs one Spec.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mveac import config as C
from mveac.calibration import baselines as B
from mveac.calibration.eac import rerank_all_general
from mveac.evaluation.maut import maut_score, select_unrestricted

SINGLE_VIEW = {"category_eac": "category", "entity_eac": "entity", "topic_eac": "topic", "sentiment_eac": "sentiment"}
UNIFORM = dict(C.MV_UNIFORM_WEIGHTS)

STATE: dict = {}


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str                      # "original" | "eac" | "baseline"
    method: str                    # e.g. "mv_eac_norm", "topic_eac", "mmr"
    lam: float = 0.0
    beta: float = 0.0
    weights: tuple = field(default_factory=tuple)   # sorted (view, weight) pairs; empty = by method
    normalize: bool = False
    ucb_mode: str = "full"
    epsilon: float = C.KL_EPSILON

    def weight_dict(self) -> dict[str, float]:
        if self.weights:
            return dict(self.weights)
        if self.method in SINGLE_VIEW:
            return {SINGLE_VIEW[self.method]: 1.0}
        return dict(UNIFORM)


def eac_spec(name, method, lam, beta, weights=None, normalize=None, ucb_mode="full", epsilon=C.KL_EPSILON) -> Spec:
    """Spec for an EAC-family configuration. ``normalize`` defaults to True for multi-view
    weights (the paper's MV-EAC) and False for a single view."""
    w = tuple(sorted((weights or ({SINGLE_VIEW[method]: 1.0} if method in SINGLE_VIEW else UNIFORM)).items()))
    if normalize is None:
        normalize = method not in SINGLE_VIEW and method != "mv_eac"
    return Spec(name, "eac", method, float(lam), float(beta), w, bool(normalize), ucb_mode, float(epsilon))


def _load_scores(path) -> tuple[dict, dict]:
    df = pd.read_parquet(path)
    counts = df.groupby("impression_id").size().to_dict()
    scores = {int(i): [(int(a), float(s)) for a, s in zip(g.article_id, g.score)]
              for i, g in df.sort_values(["impression_id", "rank"]).groupby("impression_id")}
    return scores, counts


def load_state(split: str, models: list[str], need_embeddings: bool = False, limit: int = 0) -> None:
    """Load everything one split needs into :data:`STATE`. ``models`` may include NRMS runs as
    ``nrms_<tag>`` (e.g. ``nrms_seed1``); plain ``nrms`` is the primary ``seed0`` run."""
    t0 = time.time()
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        article_maps = pickle.load(f)
    with open(C.SEM_DIR / "category_vectors.pkl", "rb") as f:
        category_vectors = pickle.load(f)
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        profiles = pickle.load(f)
    user_profiles = {v: profiles[v] for v in C.CANDIDATE_VIEWS}
    del profiles
    behaviors = pd.read_parquet(C.PROC_DIR / f"{split}_sub.parquet")
    if limit:
        keep = set(behaviors.impression_id.astype(int).tolist()[:limit])
        behaviors = behaviors[behaviors.impression_id.isin(keep)]
    scores, counts = {}, {}
    for m in models:
        scores[m], counts[m] = _load_scores(C.score_path(m, split))
        if limit:
            scores[m] = {i: c for i, c in scores[m].items() if i in keep}
    emb = {}
    if need_embeddings:
        from mveac.data.loader import load_word2vec_embeddings
        dv = load_word2vec_embeddings(C.WORD2VEC_ZIP)
        emb = {int(i): np.asarray(v, dtype=np.float32) for i, v in zip(dv.index if "article_id" not in dv.columns
                                                                         else dv["article_id"], dv["document_vector"])}
    STATE.clear()
    STATE.update(split=split, vocabs=vocabs, article_maps=article_maps, category_vectors=category_vectors,
                 user_profiles=user_profiles, behaviors=behaviors, scores=scores, counts=counts, emb=emb)
    print(f"[revision2] {split} state loaded in {time.time() - t0:.0f}s "
          f"({len(behaviors)} impressions, models={models})", flush=True)


def rerank(model: str, spec: Spec) -> dict[int, list[tuple[int, float]]]:
    """Rerank every impression of the loaded split for one base model."""
    s = STATE
    sc = s["scores"][model]
    if spec.kind == "original":
        return {int(i): c[:C.K] for i, c in sc.items()}
    if spec.kind == "baseline":
        return B.rerank_all_baseline(spec.method, spec.lam, sc, s["behaviors"], s["user_profiles"],
                                     s["article_maps"], s["vocabs"], C.K, s["emb"])
    return rerank_all_general(sc, s["behaviors"], s["user_profiles"], s["article_maps"], s["vocabs"],
                              spec.weight_dict(), spec.lam, spec.beta, C.K, epsilon=spec.epsilon,
                              normalize=spec.normalize, ucb_mode=spec.ucb_mode)


def lists_to_arrays(ranked: dict) -> tuple[np.ndarray, np.ndarray]:
    """(impression_id, K article ids padded with -1) for ``np.savez_compressed``."""
    ids = np.fromiter(ranked.keys(), dtype=np.int64, count=len(ranked))
    arts = np.full((len(ranked), C.K), -1, dtype=np.int64)
    for i, iid in enumerate(ids):
        row = [a for a, _ in ranked[iid][:C.K]]
        arts[i, :len(row)] = row
    return ids, arts


def select(grid: pd.DataFrame) -> pd.Series:
    """Revised selection rule: equal-weight MAUT over the whole grid, every beta eligible,
    ties within PARSIMONY_TOL broken by the smallest beta, then the smallest lambda."""
    g = grid.copy()
    g["maut_score"] = maut_score(g)
    return select_unrestricted(g)


def select_tradcal(grid: pd.DataFrame) -> pd.Series:
    """Trad-Cal: MAUT normalized over the whole paired grid, maximized on its beta=0 slice."""
    g = grid.copy()
    g["maut_score"] = maut_score(g)
    t = g[g.beta == 0]
    return t.loc[t["maut_score"].idxmax()]


def simplex_weights(step: int = C.SIMPLEX_STEP) -> list[dict[str, float]]:
    """All weight vectors (w_cat, w_ent, w_top) on the simplex with step 1/step."""
    out = []
    for i in range(step + 1):
        for j in range(step + 1 - i):
            out.append({"category": i / step, "entity": j / step, "topic": (step - i - j) / step})
    return out
