"""
Central configuration for the MV-EAC reproduction pipeline.

Every path in this module can be overridden with an environment variable, so the
package works whether the EB-NeRD zip files live in ``./data/raw`` (the default,
see ``docs/data.md``) or somewhere else on a shared cluster filesystem.

All hyperparameter grids, model settings and the MAUT weighting scheme reproduce
exactly the choices reported in the paper ("Multi-View Exploration-Aware
Calibration for Reducing Informational Concentration in News Recommendation").
If you change a value here, you are running a *different* experiment than the
one in the paper -- that's fine for follow-up work, just don't expect the numbers
in ``docs/pipeline.md`` to still match.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# REPO_ROOT is the mv-eac/ directory (parent of mveac/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# All raw and derived data live under DATA_ROOT. Override with the MVEAC_DATA_ROOT
# environment variable to point at a different disk (recommended: the raw EB-NeRD
# Large zip files are ~14 GB, and the processed artifacts add another ~20-30 GB).
DATA_ROOT = Path(os.environ.get("MVEAC_DATA_ROOT", REPO_ROOT / "data")).resolve()

RAW_DIR = DATA_ROOT / "raw"
EBNERD_LARGE_ZIP = RAW_DIR / "ebnerd_large.zip"
WORD2VEC_ZIP = RAW_DIR / "Ekstra_Bladet_word2vec.zip"

PROC_DIR = DATA_ROOT / "processed"
SCORES_DIR = PROC_DIR / "scores"
SEM_DIR = PROC_DIR / "semantic_spaces"
CKPT_DIR = DATA_ROOT / "checkpoints"

RESULTS_DIR = DATA_ROOT / "results"
FIGURES_DIR = DATA_ROOT / "figures"
LOGS_DIR = DATA_ROOT / "logs"

for _d in (RAW_DIR, PROC_DIR, SCORES_DIR, SEM_DIR, CKPT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Recommendation list size
# ---------------------------------------------------------------------------
K = 10  # every metric in the paper is reported at cutoff K=10

# ---------------------------------------------------------------------------
# Base recommenders
# ---------------------------------------------------------------------------
MODELS = ["most_popular", "itemknn", "nrms"]
MODEL_DISPLAY = {"most_popular": "Pop", "itemknn": "KNN", "nrms": "NRMS"}

# ---------------------------------------------------------------------------
# Stratified subsampling of the EB-NeRD Large validation behaviors
# ---------------------------------------------------------------------------
# The paper draws two disjoint stratified subsamples from the EB-NeRD Large
# *validation* split (there is no held-out test partition in the released
# dataset): 50,000 impressions for hyperparameter selection (VAL_SUBSAMPLE) and
# 500,000 impressions for the final reported evaluation (TEST_SUBSAMPLE).
# Stratification is 2-way: user history-length quartile x impression-timestamp
# quartile (4 x 4 = 16 strata), with EQUAL allocation (each stratum contributes
# 1/16 of the subsample), so cold/heavy users and early/late impressions are all
# represented -- evenly, not in population proportion.
VAL_SUBSAMPLE = 50_000
TEST_SUBSAMPLE = 500_000
N_QUANTILES = 4  # quartiles per stratification axis
RANDOM_SEED = 42
VAL_SAMPLE_SEED = RANDOM_SEED + 1
TEST_SAMPLE_SEED = RANDOM_SEED + 2

# ---------------------------------------------------------------------------
# Semantic views
# ---------------------------------------------------------------------------
# Five candidate views are extracted from EB-NeRD article metadata. Two are
# excluded from MV-EAC's retained view set (see ALL_VIEWS vs MV_VIEWS below),
# each for a different, empirically-established reason (paper Section 4.3 / RQ5):
#   - subcategory is excluded *upfront*: near-duplicate of category
#     (NMI(category, subcategory) = 0.673 on the full corpus).
#   - sentiment is evaluated as a full single-view candidate but excluded
#     *empirically*, under the paper's view-inclusion criterion (Section 5.4):
#     a view is excluded if the model without it is never worse, by a
#     non-negligible (|d| >= 0.10) effect, on any metric in any base model.
#     Adding sentiment to the 3-view model at identical (lambda, beta)
#     (SENTIMENT_ADDBACK_WEIGHTS) improves no metric non-negligibly.
ALL_VIEWS = ["category", "subcategory", "entity", "topic", "sentiment"]
CANDIDATE_VIEWS = ["category", "entity", "topic", "sentiment"]  # evaluated as single-view EAC
MV_VIEWS = ["category", "entity", "topic"]                      # MV-EAC's retained set V*
MV_UNIFORM_WEIGHTS = {v: 1.0 / len(MV_VIEWS) for v in MV_VIEWS}

ENTITY_MIN_FREQ = 10  # entities appearing in fewer articles than this are dropped from the vocab
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# The five single-view EAC methods reranked in the validation grid search
# (subcategory is excluded upfront, see ALL_VIEWS vs CANDIDATE_VIEWS above) plus
# MV-EAC itself.
GRID_METHODS = ["category_eac", "entity_eac", "topic_eac", "sentiment_eac", "mv_eac"]

# ---------------------------------------------------------------------------
# RQ2 ablation: drop one of the three retained views, redistribute its weight
# equally among the remaining two.
# ---------------------------------------------------------------------------
ABLATION_CONFIGS: dict[str, dict[str, float]] = {
    "no_category": {"entity": 0.5, "topic": 0.5},
    "no_entity": {"category": 0.5, "topic": 0.5},
    "no_topic": {"category": 0.5, "entity": 0.5},
}

# ---------------------------------------------------------------------------
# RQ4: continuous view-weight sweep (replaces a fixed set of hand-picked
# weight vectors with a systematic scan of the topic weight; category and
# entity always split the remaining weight equally). w_topic = 1/3 recovers
# MV-EAC's own uniform weighting; w_topic = 1.0 recovers Topic-EAC.
# ---------------------------------------------------------------------------
WEIGHT_SWEEP = [0.0, 0.15, 0.50, 0.70, 0.85, 1.0]


# RQ2 sentiment add-back: 4 views at 1/4, evaluated at MV-EAC's own (lambda, beta).
SENTIMENT_ADDBACK_WEIGHTS = {"category": 0.25, "entity": 0.25, "topic": 0.25, "sentiment": 0.25}


def sweep_weights(w_topic: float) -> dict[str, float]:
    """Return the 3-view weight vector for one point of the RQ4 sweep."""
    rest = (1.0 - w_topic) / 2.0
    return {"category": rest, "entity": rest, "topic": w_topic}


# ---------------------------------------------------------------------------
# KL-divergence calibration (Steck, 2018) and UCB exploration bonus
# ---------------------------------------------------------------------------
KL_EPSILON = 1e-10  # additive smoothing to keep KL(P||Q) finite when Q has zero mass

# ---------------------------------------------------------------------------
# Validation grid: (lambda, beta) hyperparameter search
# ---------------------------------------------------------------------------
LAMBDA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
# The SELECTION grid is beta <= 0.9. beta in {1.0, 1.5, 2.0} is also evaluated on
# validation, but only as a sensitivity analysis (paper Section 5.2, Table
# "beta sensitivity"); those points are never selected.
BETA_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]

# Selection is restricted to beta <= BETA_CAP, with a small parsimony tolerance
# to break ties in favor of the smallest beta, then lambda, within reach of the
# best MAUT score.
BETA_CAP = 0.9
PARSIMONY_TOL = 0.003

# ---------------------------------------------------------------------------
# MAUT (Multi-Attribute Utility Theory) hyperparameter selection
# ---------------------------------------------------------------------------
# NDCG (accuracy), Entropy (category-level diversity), ERR and TCI (entity- and
# topic-level concentration, to be minimized), weighted equally. ILD is
# deliberately excluded: on the validation grid it is near-perfectly correlated
# with Entropy (Pearson r > 0.95 for every method/model pair, since both are
# functions of the same single-label category-count vector), so including both
# would silently double-weight category diversity relative to every other
# objective.
MAUT_METRICS = ["NDCG@K", "Entropy", "ERR@K", "TCI@K"]
MAUT_MINIMIZE = {"ERR@K", "TCI@K"}
MAUT_EQUAL_WEIGHTS = {m: 0.25 for m in MAUT_METRICS}

# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------
ALPHA = 0.05
# |d_z| bands used throughout the paper. These are the authors' practical-relevance
# thresholds, deliberately lower than Cohen's (1988) 0.2/0.5/0.8 benchmarks.
EFFECT_SIZE_THRESHOLDS = {"negligible": 0.10, "small": 0.50}

# ---------------------------------------------------------------------------
# NRMS training
# ---------------------------------------------------------------------------
NRMS_INPUT_DIM = 300      # pre-trained Word2Vec document-vector dimensionality
NRMS_HIDDEN_DIM = 256
NRMS_NUM_HEADS = 4
NRMS_DROPOUT = 0.2
NRMS_MAX_HISTORY = 50
NRMS_NEG_RATIO = 4        # negatives sampled per positive during training
NRMS_EPOCHS = 5
NRMS_BATCH_SIZE = 128
NRMS_LR = 1e-3
NRMS_TRAIN_SAMPLE = 200_000   # impressions (simple random sample) from the ~12M training behaviors
# The paper's primary NRMS run ("seed0") drew its training sample with seed 42 and did
# NOT seed the parameter initialization, so it cannot be re-created bit-for-bit; its
# val/test scores are shipped under data/reference_scores/ (see docs/pipeline.md).
NRMS_PRIMARY_SAMPLE_SEED = RANDOM_SEED
NRMS_CKPT = CKPT_DIR / "nrms_seed0.pt"

# Reproducibility check (paper Section 5.3, "training-run variance"): three further
# instances, each with its own seed controlling BOTH the training sample and the
# parameter initialization (so their spread mixes both sources of variance).
NRMS_VARIANTS = {
    "seed1": {"seed": 1, "train_sample": NRMS_TRAIN_SAMPLE},
    "seed2": {"seed": 2, "train_sample": NRMS_TRAIN_SAMPLE},
    "large1M": {"seed": 0, "train_sample": 1_000_000},
}

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
N_WORKERS = int(os.environ.get("MVEAC_WORKERS", "4"))
