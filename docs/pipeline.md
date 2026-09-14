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
x impression-timestamp quartile (4x4 = 16 strata), with **equal allocation**
(each stratum contributes 1/16 of the subsample), so both subsamples cover
cold/heavy users and early/late impressions evenly -- not in population
proportion. The released EB-NeRD Large logs contain no zero-click impressions
(0 of 12,063,890 train and 0 of 12,566,385 validation behaviors), so no
click-based filtering is applied.

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
batch size 128, Adam lr=1e-3, on a 200,000-impression simple random sample of
the ~12.06M training behaviors -- see `mveac.models.nrms` for the full
architecture). Runs on CUDA, Apple Silicon MPS, or CPU, whichever is
available.

The primary run (`--tag seed0`, the default) is what every later step
reranks against. **Exact reproduction note:** the paper's primary run drew its
training sample with seed 42 (the default `--seed` here) but did not seed the
parameter initialization, so it cannot be recreated bit-for-bit. Its val/test
NRMS scores are therefore shipped in `data/reference_scores/`; copy them to
`$MVEAC_DATA_ROOT/processed/scores/scores_nrms_seed0_{val,test}.parquet` to
rerank exactly the paper's NRMS lists. The paper additionally reports a
**reproducibility check** across three further training runs, each with its
own seed controlling both the training sample and the initialization -- two
further 200,000-impression samples and one 1,000,000-impression sample.
Reproduce that check with:

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

The selection grid is beta <= 0.9; beta in {1.0, 1.5, 2.0} is evaluated too,
but only as a sensitivity analysis (Step 6).

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

Selection is restricted to **beta <= 0.9**, with a small parsimony tolerance
(0.003) that breaks ties toward the smallest (beta, lambda) within reach of the
best score. The beta > 0.9 points do not show a plateau: had they been
eligible, 9 of the 15 EAC selections would change (4 of them in lambda too),
mostly with small validation-metric differences but, for Pop/Entity-EAC, with
+0.033 NDCG@10. These alternatives are written to `beta_sensitivity.csv`
(paper Section 5.2, beta-sensitivity table). Trad-Cal's own lambda is
selected independently, from the beta=0 slice of the same grid. In both cases the
min-max normalization runs over the full 55-point grid, so the beta > 0.9 points
set the scale of the score (normalizing over the 40 eligible points only would
change the NRMS Category-EAC and Sentiment-EAC selections; paper Section 4.6).

**Runtime:** seconds (pure CSV aggregation).

---

## Step 7 -- MCF baseline (optional, `scripts/07_mcf_baseline.py`)

The Minimum-Cost-Flow calibration baseline (Abdollahpouri et al., 2023),
included as an empirical control: for the single-label views (category,
sentiment) it selects the set that exactly optimizes Trad-Cal's deterministic
objective (via network flow) rather than greedily, with no exploration term;
for the multi-label views (topic, entity) it solves a relaxation and is
reported for completeness only. Note that MCF orders its selected set with a
presentation heuristic, so on impressions with |R_u| <= K its NDCG@K can differ
from Trad-Cal's for ordering reasons alone (see `mveac.calibration.mcf`).

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
500,000-impression test subsample for every reported configuration, at the
(lambda, beta) selected in Step 6:

- **Original** (uncalibrated base ranking)
- **Trad-Cal / EAC** for each of the 4 single-view candidates + MV-EAC (10 configs)
- **RQ2 ablation**: drop one of MV-EAC's 3 retained views at a time,
  redistributing its weight equally to the other two (3 configs)
- **RQ2 sentiment add-back**: the 4 candidate views at 1/4 each, at MV-EAC's own
  (lambda, beta), so it differs from MV-EAC only in the added view (1 config)
- **RQ4 weight sweep**: `w_topic in {0, .15, .5, .7, .85, 1}` with category and
  entity always splitting the rest equally (6 configs; `w_topic = 0` is the same
  configuration as the No-Topic ablation)

For every configuration, writes one CSV row **per impression** with every
metric (this is what all statistical testing in Step 9 runs on) plus the
reranked top-K article-id lists themselves (`results/lists/*.npz`) -- so that
redefining a metric in the future never requires rerunning the reranking
itself, only recomputing metrics from the persisted lists.

**Runtime:** the second most expensive step; comparable in scale to Step 5
(24 configs x 3 models, each reranking 500,000 impressions).

---

## Step 9 -- Statistical tests (`scripts/09_statistical_tests.py`)

Runs paired Wilcoxon signed-rank tests and Cohen's d_z effect sizes (raw sign,
method_a - method_b, as in the paper's tables) across ten comparison families
forming one 465-test family (EAC vs. Trad-Cal, MCF vs. Trad-Cal, EAC vs. MCF,
MV-EAC vs. Original, ablation vs. Full MV-EAC, weight sweep vs. Full MV-EAC,
MV-EAC vs. each single view, sentiment add-back vs. MV-EAC, MV-EAC vs.
Trad-Cal-MV on the three extra NRMS runs, and the ablation/add-back replicated on those runs), each at both the impression level
and a user-clustered level (aggregating to one
paired observation per user, since the test subsample averages ~1.7
impressions per user -- non-independent observations can inflate a naive
impression-level test's apparent power). p-values are corrected for
multiplicity across the whole family using both Holm-Bonferroni and
Benjamini-Hochberg. Also writes `pooled.csv` (the descriptive pooled-over-models
effect sizes of the ablation/sweep/add-back tables; a pooled row counts as
significant only if all three per-model tests are Holm-significant) and
`ru_gtK_means.csv` (every configuration restricted to |R_u| > K).

**Runtime:** a few minutes (reads the per-impression CSVs from Step 8).

---

## Step 10 -- Figures (`scripts/10_make_figures.py`)

Produces the four result figures used in the paper (method comparison across
5 metrics, the lambda-beta heatmap, the ablation bar chart, and the weight
sweep) as vector PDFs. Pure plotting; needs Steps 5, 8, and 9's outputs.

**Runtime:** seconds.

---

## Step 11 -- Latency benchmark (optional, `scripts/11_latency_benchmark.py`)

Times only the per-impression reranking call (single process, one BLAS thread)
for every reranker at its selected hyperparameters on a random sample of test
impressions, plus the per-user cost of building the semantic profiles.

---

## Step 12 -- Greedy vs. exact calibration check (optional, `scripts/12_verify_greedy_exactness.py`)

On a random sample of test impressions, compares the greedy Trad-Cal reranker with the exact
MCF solver at the same lambda for the single-label views and reports whether MCF ever finds a
set with a higher objective value (it does not; differing sets tie in objective value).

---

## Mapping outputs to the paper

| Paper element | Produced by | File |
|---|---|---|
| Dataset statistics table | Step 1 | `data/processed/dataset_report.json` (reference copy: `data/results/dataset_report.json`) |
| Selected hyperparameters table | Steps 6, 7 | `data/results/best_params.csv`, `best_params_mcf.csv`, validation grids `val_grid_*.csv` |
| Beta-sensitivity table | Step 6 | `data/results/beta_sensitivity.csv` |
| Main results table (RQ1) | Steps 7, 8, 9 | `test_summary.csv`, `test_summary_mcf.csv`, `stats_full.csv` (families A, B, C, D) |
| |R_u| > K appendix table | Step 9 | `ru_gtK_means.csv`, `stats_full.csv` (`d_impression_ru_gtK`) |
| NRMS training-run robustness table | Steps 4, 8 (variants), 9 | `test_summary_nrms_{tag}.csv`, `stats_full.csv` (family I) |
| Ablation / add-back tables (RQ2) | Steps 8, 9 | `pooled.csv`, `stats_full.csv` (families E, H) |
| Single- vs. multi-view table (RQ3) | Steps 8, 9 | `test_summary.csv`, `stats_full.csv` (family G) |
| RQ4 weight sweep | Steps 8, 9 | `stats_full.csv` (family F), `pooled.csv` |
| Figures (method comparison, lambda-beta heatmap, ablation, weight sweep) | Step 10 | `data/figures/*.pdf` |
| Reranking latency / profile cost | Step 11 | `latency_bench.csv`, `profile_cost_bench.csv` |
| Greedy Trad-Cal = exact optimum for single-label views (Section 4.4) | Step 12 | `greedy_vs_mcf.csv` |
| NMI matrix figure | (analysis) | `nmi_matrix.csv` |
