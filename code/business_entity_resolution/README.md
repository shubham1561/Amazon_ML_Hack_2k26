# Business Entity Resolution: reproducible pipeline

Blocking with a learned ranker → GBDT pruner → GBDT matcher → GBDT re-ranker → constrained decision layer.

Produces `output/candidate_pairs.tsv` and `output/matching_results.tsv` from the raw challenge TSVs. It uses only the provided data: no external data, APIs or lookups.

**Full run instructions, requirements, timings and troubleshooting: see [`CLUSTER_RUN.md`](CLUSTER_RUN.md).**

## Quick start

```bash
pip install -r requirements.txt           # Python 3.12
python src/run_all.py --check             # verifies libraries, GPU backend, paths, data
bash run.sh                               # Linux    (Windows: powershell -File run.ps1)
```

Default layout: data in `<root>/student_resource/dataset/{train,test}`, caches in `<root>/work`, results in `<root>/output`. Override with `ER_DATA_DIR`, `ER_WORK_DIR` and `ER_OUTPUT_DIR`.

## Steps (`src/run_all.py`, each in its own process, cached, resumable with `--from <step>`)

| Step | Script | What it does |
|---|---|---|
| preprocess | `preprocess.py` | Parallel normalisation of names/addresses |
| split_countries | `data.py` | Per-country S1 / pool tables (country is an open set) |
| translit | `translit.py` | Indic-romanisation → English dictionary learned from train pairs |
| train_ranker | `train_ranker.py` | GBDT ranker that orders blocked candidates before the cap |
| pairs_train / pairs_test | `stage1_pairs.py` | 8-key blocking + ranker cap + cheap features, streamed to disk |
| stage1_fit / stage1_select_* | `stage1_model.py` | GBDT pruner → final candidate set (`candidate_pairs.tsv`) |
| submission_stage1 | `make_submission.py --score-col p1` | Early safety-net output |
| stage2_* | `stage2.py` | Rich features, 2-fold GBDT matcher, test scores |
| stage3 | `stage3.py` | Re-ranker with competition + cluster-consistency features |
| submission | `make_submission.py --score-col p3` | Thresholds tuned on OOF train, both TSVs written |
| validate | official validator | PASS / FAIL |

## Source files

| File | Role |
|---|---|
| `config.py` | Paths (env-overridable), worker count, seed |
| `normalize.py` | Name/address normalisation, phonetic skeleton |
| `preprocess.py` | Parallel normalisation + caching |
| `data.py` | Per-country tables, derived columns, transliteration, ID caches |
| `translit.py` | Learned transliteration dictionary |
| `blocking.py` | Blocking keys, vectorised joins, pre-cap, ranker cap |
| `train_ranker.py` | Ranker training data + model |
| `features.py` | Cheap and rich pair features |
| `stage1_pairs.py` | Blocking + cheap features runner (chunked, resumable) |
| `stage1_model.py` | Pruner fit / sweep / select |
| `stage2.py` | Rich + context + anchor features, matcher |
| `stage3.py` | Re-ranker |
| `models.py` | XGBoost-CUDA / LightGBM backend, group ranking utilities |
| `decide.py` | Ownership, rank-specific thresholds, vectorised exact macro-F0.5 |
| `evaluate.py` | Reference scorer |
| `make_submission.py` | Tuning + writing outputs |
| `run_all.py` | Orchestration, environment check, validation |
