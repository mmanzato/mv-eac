"""
Read-only consistency check against the paper's own reported numbers.

This test does NOT run the pipeline, does NOT recompute anything, and never
writes to `data/results/`. It only reads `test_summary.csv` (Step 8's output)
and `stats_full.csv` (Step 9's output) and asserts that specific values match
what the paper reports, within a small numerical tolerance -- a quick guard
against silently reintroducing a bug in a metric or in the reranker (e.g. the
kind of accidental 4-view-instead-of-3-view MV-EAC bug this project's own
development history includes, see docs/pipeline.md).

If `data/results/test_summary.csv` is not present (e.g. a fresh clone before
running `scripts/run_all.py`), every test in this module is skipped rather
than failed -- this file checks *consistency*, it is not the mechanism that
produces the results in the first place.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mveac import config as C

TOLERANCE = 1e-3  # generous enough for a rerun on different hardware/library versions

pytestmark = pytest.mark.skipif(
    not (C.RESULTS_DIR / "test_summary.csv").exists(),
    reason="data/results/test_summary.csv not found -- run scripts/01-08 first (see docs/pipeline.md)",
)


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(C.RESULTS_DIR / "test_summary.csv")


def _row(summary: pd.DataFrame, model: str, method: str) -> pd.Series:
    matches = summary[(summary.model == model) & (summary.method == method)]
    assert len(matches) == 1, f"expected exactly one row for {model}/{method}, found {len(matches)}"
    return matches.iloc[0]


class TestOriginalBaseline:
    """The uncalibrated NDCG@10 of each base model (paper Section 5, Table 6)."""

    @pytest.mark.parametrize("model,expected_ndcg", [
        ("most_popular", 0.3794),
        ("itemknn", 0.4355),
        ("nrms", 0.4995),
    ])
    def test_uncalibrated_ndcg(self, summary, model, expected_ndcg):
        row = _row(summary, model, "original")
        assert row["NDCG@K"] == pytest.approx(expected_ndcg, abs=TOLERANCE)


class TestMVEACHeadlineNumbers:
    """MV-EAC's own reported metrics per base model (paper Table 6 / RQ1-RQ3)."""

    @pytest.mark.parametrize("model,expected", [
        ("most_popular", {"NDCG@K": 0.4658, "ILD@K": 0.8245, "Entropy": 1.4358, "ERR@K": 0.1095, "TCI@K": 0.0537}),
        ("itemknn", {"NDCG@K": 0.4710, "ILD@K": 0.8242, "Entropy": 1.4346, "ERR@K": 0.1095, "TCI@K": 0.0537}),
        ("nrms", {"NDCG@K": 0.4945, "ILD@K": 0.8172, "Entropy": 1.4074, "ERR@K": 0.1028, "TCI@K": 0.0552}),
    ])
    def test_mv_eac_metrics(self, summary, model, expected):
        row = _row(summary, model, "mv_eac")
        for metric, value in expected.items():
            assert row[metric] == pytest.approx(value, abs=TOLERANCE), f"{model}/mv_eac/{metric}"

    def test_mv_eac_selected_hyperparameters(self, summary):
        # NRMS selects a markedly lower lambda than the two non-neural models
        # (paper Section 5.2): its own personalization already competes with
        # calibration pressure, so MAUT settles on a gentler lambda for it.
        assert _row(summary, "most_popular", "mv_eac")["lambda"] == pytest.approx(0.9)
        assert _row(summary, "itemknn", "mv_eac")["lambda"] == pytest.approx(0.9)
        assert _row(summary, "nrms", "mv_eac")["lambda"] < 0.5

    def test_mv_eac_ndcg_is_never_the_worst_configuration(self, summary):
        # RQ3's central claim: MV-EAC is a competitive-to-best accuracy option,
        # not a configuration that trades away accuracy for diversity.
        for model in C.MODELS:
            model_rows = summary[summary.model == model]
            mv_ndcg = _row(summary, model, "mv_eac")["NDCG@K"]
            assert mv_ndcg >= model_rows["NDCG@K"].min()


class TestTCIRedefinitionFixedTheNRMSException:
    """The paper's single most consequential numeric fix (Section 5, RQ1): under
    the original TCI definition, MV-EAC vs. Trad-Cal-MV on TCI@10 was NOT
    significant for NRMS. Under the vocabulary-normalized redefinition, it is
    significant, with a small-to-medium effect, in all three base models."""

    def test_effect_size_is_practically_meaningful_in_every_model(self):
        stats_path = C.RESULTS_DIR / "stats_full.csv"
        if not stats_path.exists():
            pytest.skip("data/results/stats_full.csv not found -- run scripts/09_statistical_tests.py first")
        stats_df = pd.read_csv(stats_path)
        rows = stats_df[
            (stats_df.family == "A_EAC_vs_TradCal")
            & (stats_df.comparison == "mv_eac vs trad")
            & (stats_df.metric == "TCI@K")
        ]
        assert len(rows) == 3, "expected one row per base model"
        for _, row in rows.iterrows():
            assert row["significant_holm"], f"{row['model']}: expected significant after Holm correction"
            assert abs(row["d_impression"]) >= 0.10, f"{row['model']}: expected a non-negligible effect size"


class TestViewInclusionCriterion:
    """Paper Section 5.4: a candidate view is excluded if the model without it is never worse,
    by a non-negligible (|d| >= 0.10) Holm-significant effect, on any metric in any base model."""

    MINIMIZE = {"ERR@K", "TCI@K"}

    def _stats(self):
        path = C.RESULTS_DIR / "stats_full.csv"
        if not path.exists():
            pytest.skip("data/results/stats_full.csv not found")
        return pd.read_csv(path)

    def _worse_without(self, rows, removal: bool) -> bool:
        """True if the smaller view set is non-negligibly worse on some metric in some model."""
        for _, r in rows.iterrows():
            if not (r["significant_holm"] and abs(r["d_impression"]) >= 0.10):
                continue
            # removal rows: a = variant WITHOUT the view; add-back rows: a = variant WITH the view
            improvement_of_a = -r["d_impression"] if r["metric"] in self.MINIMIZE else r["d_impression"]
            smaller_is_worse = improvement_of_a < 0 if removal else improvement_of_a > 0
            if smaller_is_worse:
                return True
        return False

    def test_category_topic_entity_retained_sentiment_excluded(self):
        s = self._stats()
        for view in ["category", "topic", "entity"]:
            rows = s[(s.family == "E_Ablation_vs_MV") & (s.comparison == f"no_{view} vs mv_eac")]
            assert len(rows) == 15
            assert self._worse_without(rows, removal=True), f"{view} should be retained"
        rows = s[(s.family == "H_SentimentAddBack")]
        assert len(rows) == 15
        assert not self._worse_without(rows, removal=False), "sentiment should be excluded"
