"""
Diversification baselines of the revised paper (Section 4.4): MMR, xQuAD, DPP and a
near-duplicate filter (Dedup).
All rerank the base recommender's candidates to K items; Rel = min-max normalized base score
per impression (same as EAC); one trade-off parameter each, selected by the same MAUT protocol.

  mmr    Carbonell & Goldstein (1998): (1-lam) Rel(i) - lam * max_{j in L} cos(e_i, e_j)
         e = EB-NeRD Word2Vec document vectors (the representation used by ItemKNN-W2V and NRMS).
  xquad  Santos et al. (2010), binary-aspect form: (1-lam) Rel(i)
         + lam * sum_a P(a|u) 1[a in X_top(i)] prod_{j in L} (1 - 1[a in X_top(j)]),
         aspects = topics, P(a|u) = the user's topic profile (uniform without history).
  dpp    Chen et al. (2018) fast greedy MAP inference; kernel L_ij = q_i cos(e_i,e_j) q_j,
         q_i = exp(alpha Rel(i)), alpha = theta / (2 (1 - theta)); remaining slots filled by
         Rel once the kernel is exhausted.
  dedup  near-duplicate filtering: in Rel order, skip a candidate whose Jaccard similarity of
         topic+entity label sets with any selected article is >= tau; skipped candidates fill
         any remaining slots in Rel order.
"""
from __future__ import annotations
import numpy as np
from mveac.calibration.traditional import minmax_normalize as _minmax_normalize


def _rel(c):
    return np.array(_minmax_normalize([s for _, s in c]), dtype=np.float64)


def _emb(ids, E, dim=300):
    X = np.stack([E.get(a, np.zeros(dim)) for a in ids]).astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def mmr(c, lam, k, E):
    ids = [a for a, _ in c]; r = _rel(c); X = _emb(ids, E); S = X @ X.T
    act = np.ones(len(ids), bool); maxsim = np.zeros(len(ids)); out = []
    for t in range(min(k, len(ids))):
        s = (1 - lam) * r - lam * (maxsim if t else 0.0)
        s[~act] = -np.inf
        b = int(np.argmax(s)); act[b] = False; out.append((ids[b], float(s[b])))
        maxsim = np.maximum(maxsim, S[b])
    return out


def xquad(c, lam, k, topic_map, tidx, p_top):
    ids = [a for a, _ in c]; r = _rel(c)
    A = np.zeros((len(ids), len(tidx)))
    for i, a in enumerate(ids):
        for t in topic_map.get(a, ()):
            j = tidx.get(t)
            if j is not None:
                A[i, j] = 1.0
    notcov = np.ones(len(tidx)); act = np.ones(len(ids), bool); out = []
    for _ in range(min(k, len(ids))):
        s = (1 - lam) * r + lam * (A @ (p_top * notcov))
        s[~act] = -np.inf
        b = int(np.argmax(s)); act[b] = False; out.append((ids[b], float(s[b])))
        notcov = notcov * (1 - A[b])
    return out


def dpp(c, theta, k, E, eps=1e-10):
    ids = [a for a, _ in c]; r = _rel(c); n = len(ids)
    X = _emb(ids, E); alpha = theta / (2 * (1 - theta))
    q = np.exp(alpha * r)
    Lk = (q[:, None] * (X @ X.T)) * q[None, :]
    cis = np.zeros((min(k, n), n)); d2 = np.diag(Lk).copy()
    sel = []; out = []
    for t in range(min(k, n)):
        cand = d2.copy(); cand[sel] = -np.inf
        b = int(np.argmax(cand))
        if cand[b] < eps:
            break
        sel.append(b); out.append((ids[b], float(np.log(cand[b]))))
        e = (Lk[b] - cis[:t].T @ cis[:t, b]) / np.sqrt(d2[b])
        cis[t] = e; d2 = d2 - e ** 2
    if len(out) < min(k, n):                      # kernel exhausted (e.g. zero vectors): fill by Rel
        rest = [i for i in np.argsort(-r, kind="stable") if i not in set(sel)]
        out += [(ids[i], float("-inf")) for i in rest[: min(k, n) - len(out)]]
    return out


def dedup(c, tau, k, topic_map, ent_map):
    ids = [a for a, _ in c]; r = _rel(c)
    order = np.argsort(-r, kind="stable")
    labs = {a: {("t", x) for x in topic_map.get(a, ())} | {("e", x) for x in ent_map.get(a, ())} for a in ids}
    sel, skipped = [], []
    for i in order:
        a = ids[i]; la = labs[a]
        dup = any(la and labs[b] and len(la & labs[b]) / len(la | labs[b]) >= tau for b in sel)
        (skipped if dup else sel).append(a)
        if len(sel) == k:
            break
    sel += skipped[: max(0, min(k, len(ids)) - len(sel))]
    return [(a, 0.0) for a in sel]


def rerank_all_baseline(method, param, base_scores, behaviors, user_profiles, item_maps, vocabs, k, E):
    i2u = dict(zip(behaviors["impression_id"].astype(int), behaviors["user_id"].astype(int)))
    tidx = {v: i for i, v in enumerate(vocabs["topic"])}
    unif = np.ones(len(vocabs["topic"])) / len(vocabs["topic"])
    res = {}
    for iid, c in base_scores.items():
        iid = int(iid); u = i2u.get(iid)
        if u is None:
            res[iid] = c[:k]; continue
        if method == "mmr":
            res[iid] = mmr(c, param, k, E)
        elif method == "xquad":
            res[iid] = xquad(c, param, k, item_maps["topic"], tidx, user_profiles["topic"].get(u, unif))
        elif method == "dpp":
            res[iid] = dpp(c, param, k, E)
        elif method == "dedup":
            res[iid] = dedup(c, param, k, item_maps["topic"], item_maps["entity"])
        else:
            raise ValueError(method)
    return res
