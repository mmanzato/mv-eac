#!/usr/bin/env python3
"""
Optional analysis -- pairwise NMI between the semantic views (paper Section 5.7.1, NMI figure).

Every article in the catalog (the union of the article maps of all views) is given one
label per view: its single label for the single-label views (category, subcategory,
sentiment) and its FIRST annotated label for the multi-label views (topic, entity).
Articles with no label under a view are pooled into a shared ``unknown`` class. NMI is
normalized by the arithmetic mean of the two labelings' entropies. This summary is used
only for the NMI analysis; reranking and evaluation keep the full multi-label structure.

Needs Step 2's semantic spaces.

Usage:
    python scripts/14_nmi_matrix.py

Output (under $MVEAC_DATA_ROOT/results/): nmi_matrix.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C

VIEWS = ["category", "subcategory", "entity", "topic", "sentiment"]


def main() -> None:
    with open(C.SEM_DIR / "article_maps.pkl", "rb") as f:
        article_maps = pickle.load(f)

    article_ids = sorted(set().union(*(article_maps[v].keys() for v in VIEWS)))
    labels = {
        v: [article_maps[v][aid][0] if article_maps[v].get(aid) else "unknown" for aid in article_ids]
        for v in VIEWS
    }

    nmi = pd.DataFrame(np.eye(len(VIEWS)), index=VIEWS, columns=VIEWS)
    for i, a in enumerate(VIEWS):
        for b in VIEWS[i + 1:]:
            value = normalized_mutual_info_score(labels[a], labels[b], average_method="arithmetic")
            nmi.loc[a, b] = nmi.loc[b, a] = value

    out = C.RESULTS_DIR / "nmi_matrix.csv"
    nmi.to_csv(out)
    print(f"{len(article_ids)} articles; "
          f"share without entity = {np.mean([x == 'unknown' for x in labels['entity']]):.3f}")
    print(nmi.round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
