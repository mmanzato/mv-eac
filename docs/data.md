# Getting the data

This project uses **EB-NeRD Large**, the large-scale Danish news
recommendation dataset released for the ACM RecSys Challenge 2024
(Kruse et al., 2024, "EB-NeRD: A Large-Scale Dataset for News Recommendation",
[doi:10.1145/3687151.3687152](https://doi.org/10.1145/3687151.3687152)).

## 1. Download

The dataset is hosted at **<https://recsys.eb.dk>**. From there:

1. Download the **EB-NeRD Large** archive (`ebnerd_large.zip`). It contains
   article metadata and the train/validation behavior and history logs.
2. Download the **Ekstra Bladet Word2Vec** article embeddings archive
   (`Ekstra_Bladet_word2vec.zip`). This ships separately from the main
   archive on the same portal, under the "pretrained embeddings" /
   "article embeddings" section.

Both are free to download but require accepting the dataset's terms of use
on the portal (a lightweight registration, no cost). Total download size is
approximately 14 GB.

## 2. Place the files

```
mv-eac/
└── data/
    └── raw/
        ├── ebnerd_large.zip
        └── Ekstra_Bladet_word2vec.zip
```

By default the pipeline looks for both files under `./data/raw/` relative to
the repository root. To use a different location (e.g. a shared cluster
filesystem), set the `MVEAC_DATA_ROOT` environment variable before running
anything:

```bash
export MVEAC_DATA_ROOT=/path/to/large/disk
mkdir -p "$MVEAC_DATA_ROOT/raw"
# place the two zip files under $MVEAC_DATA_ROOT/raw/
```

Everything derived from the raw data (semantic profiles, model scores,
reranked lists, results, figures) is also written under `$MVEAC_DATA_ROOT`,
so this one variable controls the entire pipeline's disk footprint.

**Do not extract the zip archives.** Every loader in `mveac.data.loader`
reads the parquet members directly out of the zip files; extracting them
first only doubles disk usage for no benefit.

## 3. Expected archive contents

`ebnerd_large.zip`:
```
articles.parquet             125,541 articles: category, subcategory, topics,
                              named entities, sentiment, published_time, ...
train/behaviors.parquet      ~12.06M training impressions
train/history.parquet        per-user click history recorded before the
                              training window
validation/behaviors.parquet ~12.57M impressions -- the pool this project's
                              own 50K/500K evaluation subsamples are drawn
                              from (see docs/pipeline.md, Step 1). EB-NeRD
                              Large ships no separate held-out test split.
```

`Ekstra_Bladet_word2vec.zip`:
```
Ekstra_Bladet_word2vec/document_vector.parquet
                              one pre-trained 300-d Word2Vec document
                              embedding per article_id
```

## 4. Disk space budget

| Stage | Approximate size |
|---|---|
| Raw zip archives | ~14 GB |
| Semantic profiles (`data/processed/semantic_spaces/`) | ~8-10 GB (the entity-view user profiles dominate) |
| Base-model + NRMS scores (`data/processed/scores/`) | ~2-3 GB |
| Validation grid, test evaluation, reranked lists (`data/results/`) | ~5-8 GB |

Budget roughly **35-40 GB** free disk space for a full run with the primary
NRMS instance only; add another ~10 GB per additional NRMS reproducibility
variant (`--with-nrms-variants` in `scripts/run_all.py`).
