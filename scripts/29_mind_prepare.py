#!/usr/bin/env python3
"""
Second revision, step 29 -- MIND-small as a second dataset (paper Section 5.9), prepared into the SAME structures the
EB-NeRD pipeline consumes (vocabs, article_maps, behaviors, user profiles), so that the package's
rerankers, metrics and MAUT selection run unchanged.

Source: MIND-small (Wu et al., ACL 2020), official zips mirrored by the Microsoft Recommenders
organization on Hugging Face (Recommenders/MIND). Train = impressions of 9-14 Nov 2019; dev = 15 Nov.

Design:
  * views: category (MIND "category"); topic = NMF topics (K = 80) on TF-IDF of title + abstract,
    multi-label (topics with share >= TOPIC_SHARE of the article's topic weight, at most 5; the top
    topic always kept) -- NOT the MIND subcategory; entity = WikiData ids of title + abstract
    entities, vocabulary = entities in >= 10 articles (same threshold as EB-NeRD). No sentiment view.
  * temporal split: base-model training and profile period = train impressions before 14 Nov;
    validation = random 20,000 impressions of 14 Nov; test = random 30,000 impressions of the dev
    split (15 Nov). Impressions need >= 1 click and >= 2 candidates.
  * user history = MIND "history" column of the impression's split (clicks before the data window)
    followed by the user's clicks in the training period (9-13 Nov), in time order, deduplicated.
    Users with an empty history get no profile (uniform fallback in every reranker), as in EB-NeRD.
  * article vectors for NRMS = mean of the 300-d GloVe vectors shipped in MINDsmall_utils
    (embedding_all.npy / word_dict_all.pkl) over title + abstract tokens -- the analogue of the
    frozen document vectors used on EB-NeRD.

Input: the MIND-small zips (MINDsmall_train.zip, MINDsmall_dev.zip, MINDsmall_utils.zip) unpacked under
$MVEAC_DATA_ROOT/mind/raw (see docs/data.md). Output: $MVEAC_DATA_ROOT/mind/processed/
"""
from __future__ import annotations
import json, pickle, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mveac import config as C
MIND = C.DATA_ROOT / "mind"
RAW = MIND / "raw"
OUT = MIND / "processed"
N_TOPICS, TOPIC_SHARE, TOPIC_MAX = 80, 0.15, 5
ENTITY_MIN_FREQ = 10
N_VAL, N_TEST, SEED = 20_000, 30_000, 0
VAL_DAY = pd.Timestamp("2019-11-14")
DEV_OFFSET = 10_000_000          # dev impression ids restart at 1


def nid(s: str) -> int:
    return int(s[1:])


def load_news():
    cols = ["nid", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    n = pd.concat([pd.read_csv(RAW / f"MINDsmall_{s}" / "news.tsv", sep="\t", header=None, names=cols,
                               quoting=3, keep_default_na=False) for s in ("train", "dev")])
    n = n.drop_duplicates("nid").reset_index(drop=True)
    n["article_id"] = n.nid.map(nid)
    return n


def load_beh(split):
    b = pd.read_csv(RAW / f"MINDsmall_{split}" / "behaviors.tsv", sep="\t", header=None,
                    names=["imp", "uid", "time", "history", "impressions"], keep_default_na=False)
    b["impression_id"] = b.imp.astype(int) + (DEV_OFFSET if split == "dev" else 0)
    b["user_id"] = b.uid.str[1:].astype(int)
    b["impression_time"] = pd.to_datetime(b.time, format="%m/%d/%Y %I:%M:%S %p")
    b["history_list"] = b.history.map(lambda h: [nid(x) for x in h.split()] if h else [])
    pairs = b.impressions.map(lambda s: [x.rsplit("-", 1) for x in s.split()])
    b["article_ids_inview"] = pairs.map(lambda p: [nid(a) for a, _ in p])
    b["article_ids_clicked"] = pairs.map(lambda p: [nid(a) for a, l in p if l == "1"])
    return b.drop(columns=["imp", "uid", "time", "history", "impressions"])


def topics(news):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    text = (news.title + " " + news.abstract).str.lower()
    X = TfidfVectorizer(max_features=30000, min_df=5, max_df=0.5, stop_words="english",
                        token_pattern=r"(?u)\b[a-z][a-z]+\b").fit(text)
    vocab = np.array(X.get_feature_names_out())
    M = X.transform(text)
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=SEED, max_iter=400)
    W = nmf.fit_transform(M)
    labels = []
    for w in W:
        s = w.sum()
        if s <= 0:
            labels.append([]); continue
        sh = w / s; order = np.argsort(-sh)
        keep = [int(order[0])] + [int(j) for j in order[1:TOPIC_MAX] if sh[j] >= TOPIC_SHARE]
        labels.append([f"t{j:02d}" for j in keep])
    top_words = {f"t{j:02d}": " ".join(vocab[np.argsort(-nmf.components_[j])[:8]]) for j in range(N_TOPICS)}
    return labels, top_words


def entities(news):
    def ids(js):
        try:
            return [e["WikidataId"] for e in json.loads(js)] if js else []
        except Exception:
            return []
    lists = [list(dict.fromkeys(ids(a) + ids(b))) for a, b in zip(news.title_entities, news.abstract_entities)]
    cnt = Counter(e for l in lists for e in set(l))
    vocab = sorted(e for e, c in cnt.items() if c >= ENTITY_MIN_FREQ)
    vs = set(vocab)
    return [[e for e in l if e in vs] for l in lists], vocab


def glove_vectors(news):
    class Safe(pickle.Unpickler):
        def find_class(self, module, name):   # plain dict of str -> int only
            raise pickle.UnpicklingError("globals not allowed")
    wd = Safe(open(RAW / "utils" / "word_dict_all.pkl", "rb")).load()
    E = np.load(RAW / "utils" / "embedding_all.npy", allow_pickle=False)
    tok = re.compile(r"[\w]+|[.,!?;|]")
    vecs = np.zeros((len(news), E.shape[1]), dtype=np.float32)
    for i, t in enumerate((news.title + " " + news.abstract).str.lower()):
        idx = [wd[w] for w in tok.findall(t) if w in wd]
        if idx:
            vecs[i] = E[idx].mean(0)
    return vecs


def histories(beh_split, period):
    """uid -> history column (of the split) + training-period clicks, time-ordered, deduplicated."""
    clicks = (period.sort_values("impression_time").groupby("user_id")["article_ids_clicked"]
              .agg(lambda s: [a for l in s for a in l]))
    first = beh_split.sort_values("impression_time").drop_duplicates("user_id").set_index("user_id")["history_list"]
    out = {}
    for u, h in first.items():
        seq = list(dict.fromkeys(list(h) + clicks.get(u, [])))
        if seq:
            out[int(u)] = seq
    return out


def main():
    from mveac.data.semantic_profiles import build_user_profiles as build_user_profiles_dim
    OUT.mkdir(parents=True, exist_ok=True)
    news = load_news()
    tl, top_words = topics(news)
    el, ent_vocab = entities(news)
    cat_vocab = sorted(news.category.unique().tolist())
    top_vocab = [f"t{j:02d}" for j in range(N_TOPICS)]
    aids = news.article_id.astype(int).tolist()
    maps = {"category": {a: [c] for a, c in zip(aids, news.category)},
            "topic": dict(zip(aids, tl)), "entity": dict(zip(aids, el))}
    vocabs = {"category": cat_vocab, "topic": top_vocab, "entity": ent_vocab}
    json.dump(vocabs, open(OUT / "vocabs.json", "w"))
    pickle.dump(maps, open(OUT / "article_maps.pkl", "wb"), protocol=4)
    json.dump(top_words, open(OUT / "topic_top_words.json", "w"), indent=1)
    vec = glove_vectors(news)
    pd.DataFrame({"article_id": aids, "document_vector": list(vec)}).to_parquet(OUT / "document_vector.parquet")

    tr, dev = load_beh("train"), load_beh("dev")
    period = tr[tr.impression_time < VAL_DAY]
    ok = lambda b: b[(b.article_ids_clicked.map(len) > 0) & (b.article_ids_inview.map(len) >= 2)]
    val = ok(tr[tr.impression_time >= VAL_DAY]).sample(n=N_VAL, random_state=SEED)
    test = ok(dev).sample(n=N_TEST, random_state=SEED)
    cols = ["impression_id", "user_id", "impression_time", "article_ids_inview", "article_ids_clicked"]
    period.to_parquet(OUT / "train_period.parquet")
    val[cols].to_parquet(OUT / "val_sub.parquet"); test[cols].to_parquet(OUT / "test_sub.parquet")
    for name, sub, split in [("val", val, tr), ("test", test, dev)]:
        h = histories(split[split.user_id.isin(set(sub.user_id))], period)
        pickle.dump(h, open(OUT / f"history_{name}.pkl", "wb"), protocol=4)
        prof = {d: {u: p.astype(np.float32) for u, p in build_user_profiles_dim(h, maps[d], vocabs[d]).items()}
                for d in vocabs}
        pickle.dump(prof, open(OUT / f"user_profiles_{name}.pkl", "wb"), protocol=4)
    # training-history map for NRMS (history column only, as on EB-NeRD)
    th = period.drop_duplicates("user_id").set_index("user_id")["history_list"]
    pickle.dump({int(u): list(h) for u, h in th.items() if len(h)}, open(OUT / "history_train.pkl", "wb"), protocol=4)

    rep = {"articles": len(news), "categories": len(cat_vocab), "topics": N_TOPICS, "entities": len(ent_vocab),
           "train_period_impressions": len(period), "val": len(val), "test": len(test),
           "val_users": int(val.user_id.nunique()), "test_users": int(test.user_id.nunique())}
    for d in vocabs:
        l = np.array([len(maps[d][a]) for a in aids]); rep[f"labels_per_article_{d}"] = round(float(l.mean()), 2)
        rep[f"share_no_label_{d}"] = round(float((l == 0).mean()), 3)
    for name, sub in [("val", val), ("test", test)]:
        h = pickle.load(open(OUT / f"history_{name}.pkl", "rb"))
        rep[f"{name}_cold_share"] = round(float((~sub.user_id.isin(set(h))).mean()), 3)
        n = sub.article_ids_inview.map(len); rep[f"{name}_cand_mean"] = round(float(n.mean()), 1)
        rep[f"{name}_cand_gt10"] = round(float((n > 10).mean()), 3)
    rep["share_zero_vector"] = round(float((np.abs(vec).sum(1) == 0).mean()), 4)
    json.dump(rep, open(OUT / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
