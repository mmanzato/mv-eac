"""
End-to-end integration test on a small, fully synthetic dataset shaped like
EB-NeRD (the same columns real ``mveac.data`` functions expect), exercising
the whole pipeline in miniature: semantic-profile construction, both
rerankers (single-view EAC and MV-EAC), the full metric suite, and MAUT
selection over a tiny grid.

This never touches the real dataset and never reads or writes anything
under `data/` -- it exists to catch integration bugs (a function signature
mismatch between two modules, a metric that crashes on real-shaped data)
that the narrower unit tests in test_metrics.py/test_calibration.py cannot,
without needing the ~14 GB EB-NeRD download.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mveac.calibration.eac import rerank_all
from mveac.data.semantic_profiles import (
    build_all_vocabs,
    build_article_maps,
    build_category_vectors,
    build_entity_vocab,
    build_user_profiles,
)
from mveac.evaluation.maut import maut_score, select_best_eac
from mveac.evaluation.metrics_runner import compute_all, per_impression
from mveac.models.most_popular import MostPopular


@pytest.fixture(scope="module")
def synthetic_articles() -> pd.DataFrame:
    """20 articles across 3 categories, with overlapping topics/entities/sentiment,
    shaped exactly like the columns mveac.data.semantic_profiles expects."""
    rng = np.random.default_rng(0)
    categories = ["sports", "politics", "economy"]
    topics_pool = [f"topic_{i}" for i in range(6)]
    entities_pool = [f"entity_{i}" for i in range(8)]
    sentiments = ["Negative", "Neutral", "Positive"]

    rows = []
    for aid in range(1, 21):
        rows.append({
            "article_id": aid,
            "category_str": categories[aid % len(categories)],
            "subcategory": [aid % 4],
            "ner_clusters": list(rng.choice(entities_pool, size=2, replace=False)),
            "topics": list(rng.choice(topics_pool, size=2, replace=False)),
            "sentiment_label": sentiments[aid % len(sentiments)],
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_behaviors() -> pd.DataFrame:
    """40 impressions across 10 users, each with a candidate set of 8-12 articles."""
    rng = np.random.default_rng(1)
    rows = []
    for imp_id in range(1, 41):
        uid = 1 + imp_id % 10
        n_candidates = rng.integers(8, 13)
        candidates = list(rng.choice(range(1, 21), size=n_candidates, replace=False))
        clicked = list(rng.choice(candidates, size=1, replace=False))
        rows.append({
            "impression_id": imp_id, "user_id": uid,
            "article_ids_inview": candidates, "article_ids_clicked": clicked,
            "impression_time": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=imp_id),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def semantic_spaces(synthetic_articles):
    entity_vocab = build_entity_vocab(synthetic_articles, min_freq=1)
    vocabs = build_all_vocabs(synthetic_articles, entity_min_freq=1, sentiment_labels=["Negative", "Neutral", "Positive"])
    article_maps = build_article_maps(synthetic_articles, entity_vocab=set(entity_vocab))
    category_vectors = build_category_vectors(synthetic_articles, article_maps["category"], vocabs["category"])
    return vocabs, article_maps, category_vectors


@pytest.fixture(scope="module")
def user_history(synthetic_behaviors) -> dict[int, list[int]]:
    # A simple stand-in reading history: every article a user has ever seen in-view.
    history: dict[int, list[int]] = {}
    for row in synthetic_behaviors.itertuples(index=False):
        history.setdefault(row.user_id, []).extend(row.article_ids_inview)
    return history


@pytest.fixture(scope="module")
def user_profiles(semantic_spaces, user_history):
    vocabs, article_maps, _ = semantic_spaces
    return {view: build_user_profiles(user_history, article_maps[view], vocabs[view]) for view in vocabs}


class TestSemanticProfileConstruction:
    def test_every_view_has_a_nonempty_vocabulary(self, semantic_spaces):
        vocabs, _, _ = semantic_spaces
        for view in ("category", "subcategory", "entity", "topic", "sentiment"):
            assert len(vocabs[view]) > 0, view

    def test_every_article_is_mapped_for_every_view(self, semantic_spaces, synthetic_articles):
        _, article_maps, _ = semantic_spaces
        for view, mapping in article_maps.items():
            assert set(mapping.keys()) == set(synthetic_articles["article_id"])

    def test_user_profiles_are_valid_probability_distributions(self, user_profiles):
        for view, profiles in user_profiles.items():
            for uid, vec in profiles.items():
                assert vec.sum() == pytest.approx(1.0, abs=1e-6), f"{view}/{uid}"
                assert (vec >= 0).all()


class TestBaseModelScoring:
    def test_most_popular_covers_every_impression(self, synthetic_behaviors):
        click_counts = {aid: int(count) for aid, count in
                        pd.Series([a for row in synthetic_behaviors.article_ids_clicked for a in row]).value_counts().items()}
        scores = MostPopular().fit(click_counts).score_all_impressions(synthetic_behaviors)
        assert set(scores.keys()) == set(synthetic_behaviors["impression_id"])
        for ranked in scores.values():
            assert len(ranked) > 0


class TestRerankingIntegration:
    @pytest.fixture(scope="class")
    @classmethod
    def base_scores(cls, synthetic_behaviors):
        rng = np.random.default_rng(2)
        return {
            int(row.impression_id): [(aid, float(rng.random())) for aid in row.article_ids_inview]
            for row in synthetic_behaviors.itertuples(index=False)
        }

    @pytest.mark.parametrize("method", ["original", "traditional", "category_eac", "topic_eac", "entity_eac", "mv_eac"])
    def test_every_method_produces_a_valid_top_k_list_per_impression(
        self, method, base_scores, synthetic_behaviors, user_profiles, semantic_spaces
    ):
        vocabs, article_maps, _ = semantic_spaces
        k = 5
        ranked = rerank_all(
            base_scores, synthetic_behaviors, user_profiles, article_maps, vocabs,
            method=method, lam=0.5, beta=0.5, k=k, epsilon=1e-10,
        )
        assert set(ranked.keys()) == set(synthetic_behaviors["impression_id"])
        for imp_id, top_k in ranked.items():
            original_candidates = set(aid for aid, _ in base_scores[imp_id])
            assert len(top_k) <= k
            assert len(top_k) == len(set(aid for aid, _ in top_k)), "an article was selected twice"
            assert set(aid for aid, _ in top_k) <= original_candidates

    def test_mv_eac_and_single_view_disagree_on_at_least_some_impressions(
        self, base_scores, synthetic_behaviors, user_profiles, semantic_spaces
    ):
        # A weak sanity check that the multi-view combination is not silently
        # collapsing to one of its constituent single views.
        vocabs, article_maps, _ = semantic_spaces
        mv = rerank_all(base_scores, synthetic_behaviors, user_profiles, article_maps, vocabs,
                        method="mv_eac", lam=0.7, beta=0.7, k=5)
        topic_only = rerank_all(base_scores, synthetic_behaviors, user_profiles, article_maps, vocabs,
                                method="topic_eac", lam=0.7, beta=0.7, k=5)
        differing = sum(1 for imp_id in mv if [a for a, _ in mv[imp_id]] != [a for a, _ in topic_only[imp_id]])
        assert differing > 0


class TestMetricsAndMAUTIntegration:
    def test_compute_all_returns_every_expected_metric(self, base_scores, synthetic_behaviors, user_profiles, semantic_spaces):
        vocabs, article_maps, category_vectors = semantic_spaces
        ranked = rerank_all(base_scores, synthetic_behaviors, user_profiles, article_maps, vocabs,
                            method="mv_eac", lam=0.5, beta=0.5, k=5)
        metrics = compute_all(ranked, synthetic_behaviors, article_maps, vocabs, category_vectors, k=5)
        for key in ("NDCG@K", "ILD@K", "Entropy", "ERR@K", "TCI@K"):
            assert key in metrics
            assert np.isfinite(metrics[key])

    @pytest.fixture(scope="class")
    @classmethod
    def base_scores(cls, synthetic_behaviors):
        rng = np.random.default_rng(3)
        return {
            int(row.impression_id): [(aid, float(rng.random())) for aid in row.article_ids_inview]
            for row in synthetic_behaviors.itertuples(index=False)
        }

    def test_a_tiny_maut_selection_runs_end_to_end(self, base_scores, synthetic_behaviors, user_profiles, semantic_spaces):
        vocabs, article_maps, category_vectors = semantic_spaces
        rows = []
        for lam in (0.3, 0.6, 0.9):
            for beta in (0.0, 0.5):
                ranked = rerank_all(base_scores, synthetic_behaviors, user_profiles, article_maps, vocabs,
                                    method="mv_eac", lam=lam, beta=beta, k=5)
                metrics = compute_all(ranked, synthetic_behaviors, article_maps, vocabs, category_vectors, k=5)
                rows.append({"lambda": lam, "beta": beta, **metrics})
        grid = pd.DataFrame(rows)
        grid["maut_score"] = maut_score(grid)
        best = select_best_eac(grid)
        assert best["beta"] <= 0.9
        assert {"lambda", "beta"} <= set(best.index)

    def test_per_impression_metrics_average_to_the_aggregate(self, base_scores, synthetic_behaviors, user_profiles, semantic_spaces):
        vocabs, article_maps, category_vectors = semantic_spaces
        ranked = rerank_all(base_scores, synthetic_behaviors, user_profiles, article_maps, vocabs,
                            method="topic_eac", lam=0.5, beta=0.5, k=5)
        per_imp = per_impression(ranked, synthetic_behaviors, article_maps, vocabs, category_vectors, k=5)
        aggregate = compute_all(ranked, synthetic_behaviors, article_maps, vocabs, category_vectors, k=5)
        manual_mean_ndcg = np.mean([row["NDCG@K"] for row in per_imp.values()])
        assert manual_mean_ndcg == pytest.approx(aggregate["NDCG@K"])
