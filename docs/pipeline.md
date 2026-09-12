# Reproduction pipeline

This document walks through the full pipeline, step by step, from the raw
dataset to the paper's tables and figures. Each step is a numbered script
under `scripts/`; run them in order (or use `scripts/run_all.py` to run
everything, see its `--help`). Every script is resumable -- rerunning it
after an interruption skips work it already finished.

Install the package first:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Then follow `docs/data.md` to place the raw EB-NeRD Large files under
`data/raw/` (or point `MVEAC_DATA_ROOT` elsewhere).

---

## Step 1 -- Prepare data (`scripts/01_prepare_data.py`)

EB-NeRD Large ships train/validation splits but no held-out test partition.
This project draws **two disjoint stratified subsamples from the validation
behaviors**:

- **50,000 impressions** for hyperparameter selection (`val_sub.parquet`)
- **500,000 impressions** for the final reported evaluation (`test_sub.parquet`)

Stratification is 2-way (`mveac.data.sampling`): user history-length quartile
x impression-timestamp quartile (4x4 = 16 strata), sampled proportionally
within each stratum, so both subsamples represent cold/heavy users and
early/late impressions in the same proportions as the full pool.

This step also writes `dataset_report.json`, the candidate-set-size (|R_u|)
distribution behind the paper's Table 1: because 4 of the 5 reported metrics
(ILD@K, Entropy, ERR@K, TCI@K) are order-independent set functions of the
top-K list, any impression with `|R_u| <= K` leaves every reranking method
producing the *same* set (only NDCG@K, which is order-sensitive, can still
differ). Roughly 61% of impressions fall into this category on both
subsamples -- worth knowing when interpreting effect sizes.

**Runtime:** ~10-20 minutes (dominated by reading the ~12.57M-row validation
behaviors parquet out of the zip).

---

## Step 2 -- Semantic profiles (`scripts/02_build_semantic_profiles.py`)

Builds, for each of five candidate semantic views (category, subcategory,
entity, topic, sentiment):

- a **vocabulary** (the sorted set of distinct values);
- an **article map** (`article_id -> [values]`, multi-valued for entity/topic);
- a **user profile** per view (`user_id -> normalized frequency vector` over
  that view's vocabulary, from the user's reading history) -- this is
  :math:`P_u^v` in the paper's notation.

Entities are filtered to a minimum article frequency of 10 before building
the vocabulary (`ENTITY_MIN_FREQ` in `mveac.config`), yielding ~5,852
entities. User profiles are built only for the ~250-300K users appearing in
the val/test subsamples, not the full ~790K in the dataset, to keep the
entity-view profile matrix (the largest of the five) tractable in memory
(~7 GB at that user count).

**Runtime:** ~30-60 minutes; the entity-view profile construction dominates.

---

## Step 3 -- Score base models (`scripts/03_score_base_models.py`)

Scores the val/test subsamples with the two non-neural base recommenders:

- **Most Popular**: global click count in the training behaviors.
- **ItemKNN-W2V**: cosine similarity between the mean Word2Vec embedding of
  a user's click history and each candidate's embedding.

Neither requires an iterative training loop, so this step is comparatively
fast.

**Runtime:** ~15-30 minutes.

---

## Step 4 -- Train NRMS (`scripts/04_train_nrms.py`)

Trains the third base recommender: a simplified NRMS (Wu et al., 2019)
whose news encoder is the dataset's frozen pre-trained Word2Vec embeddings,
with only the multi-head-self-attention user encoder trained (5 epochs,
batch size 128, Adam lr=1e-3, on a 200,000-impression stratified subsample of
the ~12.06M training behaviors -- see `mveac.models.nrms` for the full
architecture). Runs on CUDA, Apple Silicon MPS, or CPU, whichever is
available.

The primary run (`--tag seed0`, the default) is what every later step
reranks against. The paper additionally reports a **reproducibility check**
across three further independent training runs -- two more random seeds at
the same sample size, and one seed at a 1,000,000-impression sample -- to
confirm the reported effects are not an artifact of a single, possibly
under-trained instance. Reproduce that check with:

```bash
python scripts/04_train_nrms.py --tag seed1 --seed 1
python scripts/04_train_nrms.py --tag seed2 --seed 2
python scripts/04_train_nrms.py --tag large1M --seed 0 --train-sample 1000000
```

(each variant needs its own pass of step 8, `--models nrms --nrms-tag <tag>`).

**Runtime:** ~5-20 minutes training (hardware-dependent) + ~20-40 minutes
scoring both subsamples, per instance.

---

## Step 5 -- Validation grid search (`scripts/05_grid_search.py`)

For every (base model, method) pair -- the four single-view EAC methods
(category/entity/topic/sentiment) and MV-EAC -- reranks the 50,000-impression
validation subsample at every point of a (lambda, beta) grid:

- `lambda in {0.1, 0.3, 0.5, 0.7, 0.9}` (calibration strength)
- `beta in {0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0}` (exploration strength)

The grid is deliberately searched *past* beta=0.9 (the boundary originally
considered) to confirm the metric surface plateaus rather than being cut off
at an unexplored gradient -- see `fig_lambda_beta_heatmap.pdf` (step 10) for
the visual evidence, and `docs/pipeline.md`'s Step 6 note below for how this
is reflected in what gets reported.

This is the single most expensive step in the pipeline: 5 methods x 11 betas
x 5 lambdas x 3 base models = 825 grid points, each a full rerank of 50,000
impressions. Parallelized across `--workers` processes (fork-based, so the
shared vocabularies/profiles are loaded once per worker, not per task).

**Runtime:** several hours on a single multi-core workstation; scales
close to linearly with `--workers`. Run one (model, method) combination at a
time (`--models ... --methods ...`) to parallelize across separate machines.

---

## Step 6 -- Select hyperparameters (`scripts/06_select_hyperparameters.py`)

Reads every grid produced by Step 5 and picks one (lambda, beta) per
(model, method) via **MAUT** (Multi-Attribute Utility Theory,
`mveac.evaluation.maut`): each of NDCG@K, Entropy, ERR@K, TCI@K is min-max
normalized over the grid, the two "lower is better" metrics are inverted, and
the four are averaged with equal (0.25) weight. ILD@K is deliberately
excluded from this criterion -- on the validation grid it is near-perfectly
correlated with Entropy (Pearson r > 0.95 for every method/model pair, both
being functions of the same category-count vector), so including both would
silently double-weight category-level diversity relative to every other
objective.

The reported (lambda, beta) is capped at **beta <= 0.9**: the extended grid
from Step 5 shows every metric moves by less than 0.006 (NDCG), 0.005 (ERR),
or 0.001 (TCI) between beta=0.9 and beta=2.0, so nothing is lost by capping,
and a small parsimony tolerance (0.003) breaks ties toward the smallest
(beta, lambda) within reach of the best score. Trad-Cal's own lambda is
selected independently, from the beta=0 slice of the same grid.

**Runtime:** seconds (pure CSV aggregation).

---

## Step 7 -- MCF baseline (optional, `scripts/07_mcf_baseline.py`)

The Minimum-Cost-Flow calibration baseline (Abdollahpouri et al., 2023),
included as an empirical control: it solves the *same* deterministic
calibration objective as Trad-Cal, but exactly (via network flow) rather
than greedily, and has no exploration term. Comparing it to Trad-Cal and to
EAC isolates how much of EAC's advantage is due to exploration versus mere
optimization precision (see `mveac.calibration.mcf` for the full
documentation of the three design choices that make this a fair,
directly-comparable objective).

Requires `ortools` (already in `requirements.txt`). Run as three
subcommands, `grid` -> `select` -> `eval`, mirroring steps 5-6-8 but with a
1-D grid (lambda only -- MCF has no exploration term):

```bash
python scripts/07_mcf_baseline.py grid
python scripts/07_mcf_baseline.py select
python scripts/07_mcf_baseline.py eval
```

**Runtime:** a fraction of the EAC grid search (5 lambdas x 4 views x 3
models = 60 grid points for `grid`), but slower per-impression at `eval` time
than EAC's greedy reranker, since the flow solver runs once per impression.

---

## Step 8 -- Test-set evaluation (`scripts/08_evaluate_test.py`)

The step that actually produces the paper's headline numbers: reranks the
500,000-impression test subsample for all 24 reported configurations, at the
(lambda, beta) selected in Step 6:

- **Original** (uncalibrated base ranking)
- **Trad-Cal / EAC** for each of the 4 single-view candidates + MV-EAC (10 configs)
- **RQ2 ablation**: drop one of MV-EAC's 3 retained views at a time,
  redistributing its weight equally to the other two (3 configs)
- **RQ4 weight sweep**: `w_topic in {0, .15, .5, .7, .85, 1}` with category and
  entity always splitting the rest equally (6 configs)

For every configuration, writes one CSV row **per impression** with every
metric (this is what all statistical testing in Step 9 runs on) plus the
reranked top-K article-id lists themselves (`results/lists/*.npz`) -- so that
redefining a metric in the future never requires rerunning the reranking
itself, only recomputing metrics from the persisted lists.

**Runtime:** the second most expensive step; comparable in scale to Step 5
(24 configs x 3 models, each reranking 500,000 impressions).

---

## Step 9 -- Statistical tests (`scripts/09_statistical_tests.py`)

Runs paired Wilcoxon signed-rank tests and Cohen's d effect sizes across six
comparison families (EAC vs. Trad-Cal, MCF vs. Trad-Cal, EAC vs. MCF, MV-EAC
vs. Original, ablation vs. Full MV-EAC, weight-sweep vs. Full MV-EAC), each
at both the impression level and a user-clustered level (aggregating to one
paired observation per user, since the test subsample averages ~1.7
impressions per user -- non-independent observations can inflate a naive
impression-level test's apparent power). p-values are corrected for
multiplicity across the whole family using both Holm-Bonferroni and
Benjamini-Hochberg.

**Runtime:** a few minutes (reads the per-impression CSVs from Step 8).

---

## Step 10 -- Figures (`scripts/10_make_figures.py`)

Produces the four result figures used in the paper (method comparison across
5 metrics, the lambda-beta heatmap, the ablation bar chart, and the weight
sweep) as vector PDFs. Pure plotting; needs Steps 5, 8, and 9's outputs.

**Runtime:** seconds.

---

## Mapping outputs to the paper

| Paper element | Produced by | File |
|---|---|---|
| Table 1 (dataset statistics) | Step 1 | `data/processed/dataset_report.json` |
| Table 5 (selected hyperparameters) | Step 6 | `data/results/best_params.csv` |
| Table 6 (main results, RQ1) | Step 8 + 9 | `data/results/test_summary.csv`, `stats_full.csv` (family A) |
| Table 7/8 (ablation, RQ2) | Step 8 + 9 | `stats_full.csv` (family E) |
| Table 9 (single- vs. multi-view, RQ3) | Step 8 | `test_summary.csv` |
| RQ4 weight sweep | Step 8 + 9 | `stats_full.csv` (family F) |
| Figure "method comparison" | Step 10 | `data/figures/fig_method_comparison.pdf` |
| Figure "lambda-beta heatmap" | Step 10 | `data/figures/fig_lambda_beta_heatmap.pdf` |
| Figure "ablation" | Step 10 | `data/figures/fig_ablation.pdf` |
| Figure "weight sweep" | Step 10 | `data/figures/fig_weight_sweep.pdf` |
| NRMS reproducibility check | Steps 4 + 8 (variants) | `test_summary_nrms_{tag}.csv` |
