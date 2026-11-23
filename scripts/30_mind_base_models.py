#!/usr/bin/env python3
"""
Second revision, step 30 -- base recommenders on MIND-small (Most Popular and NRMS; paper Section 5.9).

  most_popular: click counts over the training period (impressions before 14 Nov), as on EB-NeRD.
  nrms:         mveac.models.nrms.NRMS with the EB-NeRD hyper-parameters (frozen 300-d article vectors,
                hidden 256, 4 heads, dropout 0.2, neg_ratio 4, max_history 50, NRMS_EPOCHS / NRMS_BATCH_SIZE /
                NRMS_LR, seed 0), trained on the training-period impressions of users with a history.

Writes scores_{model}_{val,test}.parquet (impression_id, article_id, rank, score) and the base-model
accuracy (AUC, MRR, NDCG@10 averaged over impressions) to base_accuracy.csv.
"""
from __future__ import annotations
import pickle, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
MIND = C.DATA_ROOT / "mind"
P = MIND / "processed"
S = MIND / "scores"; S.mkdir(parents=True, exist_ok=True)


def save(scores, model, split):
    rows = [(i, a, r, s) for i, rk in scores.items() for r, (a, s) in enumerate(rk, 1)]
    pd.DataFrame(rows, columns=["impression_id", "article_id", "rank", "score"]).to_parquet(S / f"scores_{model}_{split}.parquet")


def accuracy(scores, sub):
    from sklearn.metrics import roc_auc_score
    clicks = dict(zip(sub.impression_id, sub.article_ids_clicked))
    auc, mrr, ndcg = [], [], []
    for i, rk in scores.items():
        c = set(clicks[i]); y = np.array([a in c for a, _ in rk], float)
        s = np.array([x for _, x in rk], float)
        if np.isnan(s).all():
            s = np.zeros_like(s)
        s = np.nan_to_num(s, nan=np.nanmin(s) - 1 if np.isfinite(s).any() else 0)
        if 0 < y.sum() < len(y):
            auc.append(roc_auc_score(y, s))
        mrr.append(1 / (np.argmax(y) + 1))
        g = (y[:10] / np.log2(np.arange(2, min(10, len(y)) + 2))).sum()
        ideal = (1 / np.log2(np.arange(2, min(int(y.sum()), 10) + 2))).sum()
        ndcg.append(g / ideal)
    return np.mean(auc), np.mean(mrr), np.mean(ndcg)


def main():
    import torch
    from mveac.models.most_popular import MostPopular as NewsMostPopular
    from mveac.models.nrms import NRMS

    period = pd.read_parquet(P / "train_period.parquet")
    subs = {s: pd.read_parquet(P / f"{s}_sub.parquet") for s in ("val", "test")}
    acc = []

    counts = {}
    for l in period.article_ids_clicked:
        for a in l:
            counts[int(a)] = counts.get(int(a), 0) + 1
    mp_ = NewsMostPopular().fit(counts)
    for s, sub in subs.items():
        sc = mp_.score_all_impressions(sub); save(sc, "most_popular", s)
        acc.append(("most_popular", s, *accuracy(sc, sub)))

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dv = pd.read_parquet(P / "document_vector.parquet")
    emb = np.stack(dv.document_vector.values).astype(np.float32)
    id2idx = {int(a): i for i, a in enumerate(dv.article_id)}
    hist_train = pickle.load(open(P / "history_train.pkl", "rb"))
    model = NRMS(input_dim=emb.shape[1], hidden_dim=C.NRMS_HIDDEN_DIM, num_heads=C.NRMS_NUM_HEADS,
                 dropout=C.NRMS_DROPOUT, epochs=C.NRMS_EPOCHS, batch_size=C.NRMS_BATCH_SIZE, lr=C.NRMS_LR,
                 neg_ratio=C.NRMS_NEG_RATIO, max_history=C.NRMS_MAX_HISTORY, seed=0,
                 checkpoint_path=S / "nrms_mind_s0.pt")
    # an all-padding history makes the NRMS attention softmax NaN (every position masked) and poisons the
    # weights; on EB-NeRD every training user had a history, so train on users with a history (same condition)
    fit_beh = period[period.user_id.isin(set(hist_train))]
    print(f"NRMS training impressions: {len(fit_beh)} of {len(period)} (users with history)", flush=True)
    t0 = time.time(); model.fit(fit_beh, hist_train, emb, id2idx); print(f"NRMS trained in {(time.time()-t0)/60:.1f} min", flush=True)
    for s, sub in subs.items():
        h = pickle.load(open(P / f"history_{s}.pkl", "rb"))
        sc = model.score_all_impressions(sub, h); save(sc, "nrms", s)
        acc.append(("nrms", s, *accuracy(sc, sub)))
    df = pd.DataFrame(acc, columns=["model", "split", "AUC", "MRR", "NDCG@10"])
    df.to_csv(S / "base_accuracy.csv", index=False); print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
