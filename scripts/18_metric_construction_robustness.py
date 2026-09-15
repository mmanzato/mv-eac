#!/usr/bin/env python3
"""
Optional analysis -- robustness of ERR@K and TCI@K to their construction (Supplementary
Section S7). Uses the persisted test-set top-K lists of step 8 (results/lists/*.npz); no reranking.

Per list: ERR@K as in the paper (entity mentions with multiplicity), a cross-article ERR@K
(each article contributes its set of entities), the number of entity-bearing articles and of
entity mentions, distinct entities per article, TCI@K as in the paper, and a TCI@K in which each
article distributes a total topic mass of 1 over its labels. Reports means per method and the
paired Cohen's d of the key comparisons under each variant.

Usage:
    python scripts/18_metric_construction_robustness.py [--workers 8]

Output (under $MVEAC_DATA_ROOT/results/):
    metric_construction_means.csv     one row per (model, method)
    metric_construction_effects.csv   one row per (model, comparison, variant): Cohen's d
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C
from mveac.evaluation.stats import cohens_d

METHODS = ["original", "traditional_entity_eac", "entity_eac", "traditional_topic_eac", "topic_eac",
           "category_eac", "traditional_mv_eac", "mv_eac", "ablation_no_entity", "ablation_no_topic",
           "ablation_no_category"]
COMPARISONS = [("entity_eac", "traditional_entity_eac"), ("topic_eac", "traditional_topic_eac"),
               ("mv_eac", "traditional_mv_eac"), ("mv_eac", "topic_eac"), ("mv_eac", "original"),
               ("ablation_no_entity", "mv_eac"), ("ablation_no_topic", "mv_eac")]
VARIANTS = ["ERR", "ERR_cross_article", "entity_articles", "entity_mentions", "entities_per_article",
            "TCI", "TCI_article_mass"]
_SHARED: dict = {}


def _tci(freq: dict[str, float], n_vocab: int) -> float:
    total = sum(freq.values())
    if total == 0:
        return 0.0
    hhi = sum((f / total) ** 2 for f in freq.values())
    return min(1.0, max(0.0, (hhi - 1 / n_vocab) / (1 - 1 / n_vocab)))


def _one(task):
    model, method = task
    data = np.load(C.RESULTS_DIR / "lists" / f"{model}_{method}.npz")
    ent, top, n_top = _SHARED["ent"], _SHARED["top"], _SHARED["n_top"]
    rows = []
    for iid, arts in zip(data["impression_id"], data["articles"]):
        arts = [int(a) for a in arts if a >= 0]
        mentions = [e for a in arts for e in ent.get(a, ())]
        sets = [set(ent.get(a, ())) for a in arts]
        n_set = sum(len(s) for s in sets)
        union = set().union(*sets) if sets else set()
        counts, mass = {}, {}
        for a in arts:
            labels = top.get(a, ())
            for x in labels:
                counts[x] = counts.get(x, 0) + 1
                mass[x] = mass.get(x, 0) + 1.0 / len(labels)
        rows.append((int(iid),
                     (len(mentions) - len(union)) / len(mentions) if mentions else 0.0,
                     (n_set - len(union)) / n_set if n_set else 0.0,
                     sum(1 for s in sets if s), len(mentions),
                     len(union) / len(arts) if arts else 0.0,
                     _tci(counts, n_top), _tci(mass, n_top)))
    return task, pd.DataFrame(rows, columns=["impression_id"] + VARIANTS).set_index("impression_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    args = ap.parse_args()
    maps = pickle.load(open(C.SEM_DIR / "article_maps.pkl", "rb"))
    vocabs = json.load(open(C.SEM_DIR / "vocabs.json"))
    _SHARED.update(ent={int(k): v for k, v in maps["entity"].items() if v},
                   top={int(k): v for k, v in maps["topic"].items() if v}, n_top=len(vocabs["topic"]))
    tasks = [(m, meth) for m in C.MODELS for meth in METHODS]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        per = dict(pool.map(_one, tasks))

    means = pd.DataFrame([{"model": m, "method": meth, **per[(m, meth)].mean().to_dict()} for m, meth in tasks])
    means.to_csv(C.RESULTS_DIR / "metric_construction_means.csv", index=False)
    effects = []
    for m in C.MODELS:
        for a, b in COMPARISONS:
            da, db = per[(m, a)], per[(m, b)].loc[per[(m, a)].index]
            for var in VARIANTS:
                effects.append({"model": m, "comparison": f"{a} vs {b}", "variant": var,
                                "d": cohens_d((da[var] - db[var]).to_numpy())})
    effects = pd.DataFrame(effects)
    effects.to_csv(C.RESULTS_DIR / "metric_construction_effects.csv", index=False)
    print(effects.pivot_table(index=["comparison", "model"], columns="variant", values="d").round(3).to_string())


if __name__ == "__main__":
    main()
