"""Unit tests for every metric in mveac.metrics, against small hand-computed examples.

None of these tests touch the dataset or any pipeline output -- they exist to
pin down the exact formulas (so a future refactor can't silently change one)
independently of whether EB-NeRD has even been downloaded.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mveac.metrics.accuracy import ndcg_at_k
from mveac.metrics.concentration import entity_repetition_rate, herfindahl_index, topic_concentration_index
from mveac.metrics.diversity import intra_list_diversity, jaccard_ild, list_entropy


class TestNDCG:
    def test_perfect_ranking(self):
        # The one clicked article is in first place -> NDCG = 1.0 regardless of K.
        assert ndcg_at_k([10, 20, 30], clicked_ids={10}, k=3) == pytest.approx(1.0)

    def test_clicked_article_in_second_place(self):
        dcg = 1.0 / math.log2(3)   # position 2 (0-indexed rank 1)
        idcg = 1.0 / math.log2(2)  # one relevant item, ideally at position 1
        assert ndcg_at_k([10, 20, 30], clicked_ids={20}, k=3) == pytest.approx(dcg / idcg)

    def test_no_clicks_returns_zero(self):
        assert ndcg_at_k([10, 20, 30], clicked_ids=set(), k=3) == 0.0

    def test_clicked_article_outside_top_k_scores_zero(self):
        assert ndcg_at_k([10, 20, 30], clicked_ids={30}, k=1) == 0.0


class TestIntraListDiversity:
    def test_identical_vectors_have_zero_distance(self):
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}
        assert intra_list_diversity([1, 2], vectors, k=2) == pytest.approx(0.0)

    def test_orthogonal_vectors_have_unit_distance(self):
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        assert intra_list_diversity([1, 2], vectors, k=2) == pytest.approx(1.0)

    def test_mixed_example(self):
        # article 1 and 3 share the same category vector; article 2 is orthogonal
        # to both -- mean distance over the 3 pairs is (1 + 0 + 1) / 3.
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([1.0, 0.0])}
        assert intra_list_diversity([1, 2, 3], vectors, k=3) == pytest.approx(2 / 3)

    def test_single_item_list_is_zero(self):
        vectors = {1: np.array([1.0, 0.0])}
        assert intra_list_diversity([1], vectors, k=5) == 0.0


class TestEntropy:
    def test_uniform_two_way_split_matches_ln2(self):
        article_map = {1: ["a"], 2: ["b"]}
        assert list_entropy([1, 2], article_map, vocab=["a", "b"], k=2) == pytest.approx(math.log(2))

    def test_single_value_has_zero_entropy(self):
        article_map = {1: ["a"], 2: ["a"], 3: ["a"]}
        assert list_entropy([1, 2, 3], article_map, vocab=["a", "b"], k=3) == 0.0

    def test_two_to_one_split(self):
        article_map = {1: ["a"], 2: ["b"], 3: ["a"]}
        expected = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
        assert list_entropy([1, 2, 3], article_map, vocab=["a", "b"], k=3) == pytest.approx(expected)


class TestJaccardILD:
    def test_identical_label_sets_have_zero_distance(self):
        article_map = {1: ["a", "b"], 2: ["a", "b"]}
        assert jaccard_ild([1, 2], article_map, k=2) == pytest.approx(0.0)

    def test_disjoint_label_sets_have_unit_distance(self):
        article_map = {1: ["a"], 2: ["b"]}
        assert jaccard_ild([1, 2], article_map, k=2) == pytest.approx(1.0)

    def test_partial_overlap(self):
        article_map = {1: ["a"], 2: ["a", "b"], 3: ["b"]}
        # (1,2): |{a}|/|{a,b}|=0.5 -> dist 0.5 | (1,3): disjoint -> dist 1.0
        # (2,3): |{b}|/|{a,b}|=0.5 -> dist 0.5
        assert jaccard_ild([1, 2, 3], article_map, k=3) == pytest.approx((0.5 + 1.0 + 0.5) / 3)


class TestEntityRepetitionRate:
    def test_all_unique_entities_scores_zero(self):
        entity_map = {1: ["e1"], 2: ["e2"], 3: ["e3"]}
        assert entity_repetition_rate([1, 2, 3], entity_map, k=3) == 0.0

    def test_full_repetition_approaches_one(self):
        entity_map = {1: ["e1"], 2: ["e1"], 3: ["e1"]}
        assert entity_repetition_rate([1, 2, 3], entity_map, k=3) == pytest.approx(2 / 3)

    def test_mixed_repetition(self):
        # exposures = [e1, e2, e1, e3] -> 4 total, 3 unique -> ERR = 1/4
        entity_map = {1: ["e1", "e2"], 2: ["e1"], 3: ["e3"]}
        assert entity_repetition_rate([1, 2, 3], entity_map, k=3) == pytest.approx(0.25)

    def test_no_entities_scores_zero(self):
        assert entity_repetition_rate([1, 2], {}, k=2) == 0.0


class TestTopicConcentrationIndex:
    def test_single_topic_list_is_maximally_concentrated(self):
        # The old (superseded) formula was 0/0 here; the vocabulary-normalized
        # version must return exactly 1.0 -- this is the case the redefinition
        # in the paper (Section 4.5) was specifically designed to fix.
        topic_map = {1: ["t1"], 2: ["t1"]}
        assert topic_concentration_index([1, 2], topic_map, topic_vocab_size=4, k=2) == pytest.approx(1.0)

    def test_perfectly_uniform_over_the_full_vocabulary(self):
        topic_map = {1: ["t1"], 2: ["t2"], 3: ["t3"], 4: ["t4"]}
        assert topic_concentration_index([1, 2, 3, 4], topic_map, topic_vocab_size=4, k=4) == pytest.approx(0.0)

    def test_sensitive_to_how_many_topics_a_list_reaches(self):
        # Two lists both evenly spread across their own present topics score
        # DIFFERENTLY once normalized against the full vocabulary -- reaching
        # only 2 of 8 vocabulary topics is more concentrated than reaching 4 of 8,
        # even though each list is internally perfectly even. This is exactly the
        # blind spot the paper's TCI@K redefinition fixes (the old n-normalized
        # formula scored both lists identically).
        two_topics = {1: ["t1"], 2: ["t2"]}
        four_topics = {1: ["t1"], 2: ["t2"], 3: ["t3"], 4: ["t4"]}
        tci_two = topic_concentration_index([1, 2], two_topics, topic_vocab_size=8, k=2)
        tci_four = topic_concentration_index([1, 2, 3, 4], four_topics, topic_vocab_size=8, k=4)
        assert tci_two > tci_four

    def test_no_topics_present_scores_zero(self):
        assert topic_concentration_index([1, 2], {}, topic_vocab_size=10, k=2) == 0.0

    def test_result_is_always_clipped_to_unit_interval(self):
        topic_map = {i: [f"t{i}"] for i in range(20)}
        value = topic_concentration_index(list(range(20)), topic_map, topic_vocab_size=78, k=10)
        assert 0.0 <= value <= 1.0


class TestHerfindahlIndex:
    def test_matches_hand_computed_value(self):
        topic_map = {1: ["t1"], 2: ["t1"], 3: ["t2"]}
        # shares = [2/3, 1/3] -> HHI = 4/9 + 1/9 = 5/9
        assert herfindahl_index([1, 2, 3], topic_map, k=3) == pytest.approx(5 / 9)

    def test_does_not_depend_on_vocabulary_size(self):
        # Unlike TCI@K, the raw HHI has no vocabulary-size argument at all --
        # it is a property of the list's own topic distribution only.
        topic_map = {1: ["t1"], 2: ["t2"]}
        assert herfindahl_index([1, 2], topic_map, k=2) == pytest.approx(0.5)
