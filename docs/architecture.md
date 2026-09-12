# Code layout

```
mv-eac/
├── mveac/                     installable package -- the actual algorithms
│   ├── config.py              paths, hyperparameter grids, all tunable settings
│   ├── data/
│   │   ├── loader.py           read EB-NeRD Large parquet files out of the raw zips
│   │   ├── sampling.py         stratified val/test subsample construction
│   │   └── semantic_profiles.py vocabularies, article maps, user/global profiles
│   ├── models/
│   │   ├── most_popular.py
│   │   ├── item_knn.py
│   │   └── nrms.py             the three base recommenders
│   ├── calibration/
│   │   ├── traditional.py      shared KL-divergence primitives
│   │   ├── eac.py              Traditional calibration, single/multi-view EAC,
│   │   │                       and rerank_all() -- the one dispatch function every
│   │   │                       pipeline script calls to rerank a set of impressions
│   │   └── mcf.py              the Minimum-Cost-Flow baseline (ortools)
│   ├── metrics/
│   │   ├── accuracy.py         NDCG@K
│   │   ├── diversity.py        ILD@K, Entropy@K, Jaccard-ILD
│   │   └── concentration.py    ERR@K, TCI@K (the paper's two proposed metrics)
│   └── evaluation/
│       ├── metrics_runner.py   aggregation glue: per-impression + mean-over-impressions
│       ├── maut.py             MAUT hyperparameter selection
│       └── stats.py            Wilcoxon, Cohen's d, Holm-Bonferroni / BH correction
├── scripts/                   the numbered, resumable pipeline (see docs/pipeline.md)
├── tests/                     unit tests + a check against the paper's own numbers
└── docs/                      this file, data.md, pipeline.md
```

## Design principles

- **One dispatch function.** Every rerank in the whole pipeline -- validation
  grid search, test evaluation, the RQ2 ablation, the RQ4 weight sweep --
  goes through `mveac.calibration.eac.rerank_all`. The RQ2/RQ4 variants are
  not special-cased reranking code; they are just different `weights` dicts
  passed to the same multi-view reranker.
- **Metrics operate on plain Python/numpy, not on a bespoke class hierarchy.**
  A ranked list is a `list[(article_id, score)]`; an article map is a
  `dict[article_id, list[str]]`; a profile is a `numpy.ndarray`. This keeps
  every metric function testable in isolation with a five-line synthetic
  example (see `tests/test_metrics.py`).
- **Per-impression first, aggregate second.** `mveac.evaluation.metrics_runner`
  always computes one row per impression; the validation-grid mean is a
  reduction over that, not a separately implemented code path. This is what
  makes the paired statistical tests in `mveac.evaluation.stats` possible --
  they need the same per-impression values, not just a summary.
- **Config is data, not code.** Every hyperparameter grid, weight vector, and
  file path lives in `mveac/config.py`. Changing an experiment (e.g. a
  different beta grid) never requires touching the algorithm modules.
