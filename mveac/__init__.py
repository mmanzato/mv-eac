"""
mveac -- Multi-View Exploration-Aware Calibration for news recommendation.

Reference implementation accompanying the paper "Multi-View Exploration-Aware
Calibration for Reducing Informational Concentration in News Recommendation".
See ``docs/pipeline.md`` in the repository root for the full, step-by-step
reproduction guide, and ``README.md`` for a quick start.

Package layout:
    mveac.config              central paths, hyperparameter grids, and settings
    mveac.data                EB-NeRD loading, stratified sampling, semantic profiles
    mveac.models              base recommenders (Most Popular, ItemKNN, NRMS)
    mveac.calibration.traditional   shared KL-divergence primitives
    mveac.calibration.eac           Traditional calibration, single/multi-view EAC,
                                     and rerank_all() -- the single dispatch entry
                                     point used by every pipeline script
    mveac.calibration.mcf           the Minimum-Cost-Flow baseline
    mveac.metrics             NDCG, ILD, Entropy, ERR@K, TCI@K
    mveac.evaluation          MAUT hyperparameter selection and statistical testing
"""
from __future__ import annotations

__version__ = "1.1.0"
