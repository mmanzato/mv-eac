#!/usr/bin/env python3
"""
Second revision, step 25 -- accuracy of the base recommenders (paper Table 2), computed from the
base scores over ALL candidates of each test impression (not the top-K list).

AUC per impression with >= 1 click and >= 1 non-click (average ranks for ties; NaN scores, e.g. NRMS
for a user without history, are set to the impression minimum, so an all-NaN impression has AUC 0.5);
MRR over the full candidate ranking; NDCG@10 of the full ranking. Reported for all impressions and for
users with history.

Output: results/rev2/base_model_accuracy.csv, results/rev2/base_model_accuracy_per_impression.parquet
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mveac import config as C

OUT = C.RESULTS_DIR / "rev2"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(C.PROC_DIR / "test_sub.parquet", columns=["impression_id", "user_id", "article_ids_clicked"])
    test = test.drop_duplicates("impression_id")
    clicks = {int(i): set(int(x) for x in c) for i, c in zip(test.impression_id, test.article_ids_clicked)}
    with open(C.SEM_DIR / "user_profiles.pkl", "rb") as f:
        prof = pickle.load(f)["category"]
    warm = {int(i): int(u) in prof for i, u in zip(test.impression_id, test.user_id)}
    rows = []
    for model in C.MODELS:
        sc = pd.read_parquet(C.score_path(model, "test"), columns=["impression_id", "article_id", "score"])
        for iid, g in sc.groupby("impression_id", sort=False):
            iid = int(iid)
            y = g.article_id.isin(clicks[iid]).values.astype(int)
            s = g.score.values.astype(float)
            s = np.zeros_like(s) if np.isnan(s).all() else np.where(np.isnan(s), np.nanmin(s), s)
            npos, nneg = y.sum(), len(y) - y.sum()
            if npos == 0 or nneg == 0:
                continue
            r = rankdata(s)
            auc = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            pos = np.where(y[np.argsort(-s, kind="stable")] == 1)[0]
            dcg = sum(1 / math.log2(p + 2) for p in pos if p < 10)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(npos, 10)))
            rows.append((model, iid, warm[iid], auc, 1.0 / (pos[0] + 1), dcg / idcg))
    df = pd.DataFrame(rows, columns=["model", "impression_id", "warm", "AUC", "MRR", "NDCG10_full_ranking"])
    df.to_parquet(OUT / "base_model_accuracy_per_impression.parquet", index=False)
    cols = ["AUC", "MRR", "NDCG10_full_ranking"]
    s = pd.concat([df.groupby("model")[cols].mean().assign(subset="all"),
                   df[df.warm].groupby("model")[cols].mean().assign(subset="with_history")])
    s.to_csv(OUT / "base_model_accuracy.csv")
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
