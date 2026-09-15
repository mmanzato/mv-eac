# MV-EAC: Multi-View Exploration-Aware Calibration

Reference implementation and full reproduction pipeline for **"Multi-View
Exploration-Aware Calibration for Reducing Informational Concentration in
News Recommendation"**.

News recommenders can repeatedly expose users to the same entities and
topics even when a recommendation list looks diverse at the category level.
This project introduces:

- **Exploration-Aware Calibration (EAC)**: a post-processing reranker that
  augments KL-divergence calibration (Steck, 2018) with a UCB-style
  exploration bonus.
- **Multi-View EAC (MV-EAC)**: applies EAC jointly across three complementary
  semantic views -- category, entity, and topic.
- **Two new metrics**: Entity Repetition Rate (**ERR@K**) and a Topic
  Concentration Index (**TCI@K**), designed to catch informational
  concentration that category-level diversity metrics miss.

Evaluated on **EB-NeRD Large** with three base recommenders (Most Popular,
ItemKNN, NRMS), MV-EAC is a balanced default rather than a uniformly
dominant method: its exploration term lowers topic concentration beyond
deterministic multi-view calibration in every base model at negligible accuracy
cost, and against the strongest single view (Topic-EAC) it is tied on accuracy
and entity repetition, higher on category-level diversity, and slightly higher
on topic concentration.

## Quick start

```bash
git clone <this-repository-url> mv-eac
cd mv-eac
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Get the data (see docs/data.md) -- place the two EB-NeRD Large zip files
#    under data/raw/, or point MVEAC_DATA_ROOT elsewhere.

# 2. Run the full pipeline (see docs/pipeline.md for what each step does and
#    how long to expect it to take -- this is a multi-hour run against the
#    full dataset, dominated by the validation grid search and test
#    evaluation steps).
python scripts/run_all.py

# 3. (optional) also run the MCF baseline and the NRMS reproducibility check
python scripts/run_all.py --with-mcf --with-nrms-variants

# 4. (optional) reranking latency / profile-construction cost
python scripts/11_latency_benchmark.py

# 5. (optional) greedy-vs-exact check, epsilon sensitivity, and view NMI
python scripts/12_verify_greedy_exactness.py
python scripts/13_eps_sensitivity.py
python scripts/14_nmi_matrix.py

# 6. (optional) view selection on the validation set (Section 5.4)
python scripts/15_validation_view_selection.py

# 7. (optional) robustness analyses of the Supplementary Material (Section S7)
python scripts/11_latency_benchmark.py --min-candidates 11
python scripts/16_view_term_scales.py
python scripts/17_unlabeled_article_rule.py
python scripts/18_metric_construction_robustness.py
```

Every script under `scripts/` is independently runnable and resumable --
see `docs/pipeline.md` for the full step-by-step breakdown, expected runtime
per step, and a table mapping every output file to the paper's tables and
figures.

## Running the tests

```bash
pytest tests/ -v
```

The test suite is split into two kinds:

- **Unit tests** (`test_metrics.py`, `test_calibration.py`, `test_maut.py`,
  `test_stats.py`) check the core formulas against small, hand-computed
  synthetic examples. These run in well under a second and need no data
  download.
- **Paper-consistency check** (`test_paper_consistency.py`) is a read-only
  sanity check against the actual result files this project's own pipeline
  produced -- it asserts that specific numbers reported in the paper (e.g.
  MV-EAC's NDCG@10 for the Most Popular base model) are still what
  `data/results/test_summary.csv` contains, within a small numerical
  tolerance. It is skipped automatically if that file is not present rather
  than failing -- it never regenerates or modifies any result file. A small
  set of reference result CSVs (tens of KB, not the dataset itself) ships with
  the repository under `data/results/`. These files are the outputs of the
  research pipeline that produced the paper's numbers; the check therefore
  verifies that a fresh run of *this* code reproduces them (after running the
  pipeline, compare your `data/results/` against the shipped copies), not that
  the shipped copies are self-consistent. The primary NRMS run's scores, which
  cannot be regenerated bit-for-bit, are provided under `data/reference_scores/`.

## Repository layout

See `docs/architecture.md` for the full package layout and design
principles. In short: `mveac/` is the installable library (data loading,
models, calibration/EAC/MCF, metrics, MAUT and statistical evaluation);
`scripts/` is the numbered, resumable reproduction pipeline; `tests/` and
`docs/` are what their names say.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@article{ferraz2026mveac,
  title   = {Multi-View Exploration-Aware Calibration for Reducing Informational Concentration in News Recommendation},
  author  = {Ferraz, Carolina Toledo and Manzato, Marcelo Garcia},
  journal = {Information Processing \& Management},
  year    = {2026}
}
```

## License

MIT -- see `LICENSE`.
