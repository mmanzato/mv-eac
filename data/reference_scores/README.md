# Reference NRMS scores (primary run, "seed0")

`scores_nrms_seed0_{val,test}.parquet` are the NRMS relevance scores behind every
NRMS number in the paper's main tables (columns: `impression_id`, `article_id`,
`rank`, `score`; one row per candidate article of the 50,000 validation and
500,000 test impressions).

They are shipped because that training run drew its 200,000-impression training
sample with seed 42 but did not seed the parameter initialization, so retraining
cannot reproduce these exact scores. To rerank exactly the paper's NRMS lists,
copy both files to `$MVEAC_DATA_ROOT/processed/scores/` and skip the default
(`--tag seed0`) run of `scripts/04_train_nrms.py`.
