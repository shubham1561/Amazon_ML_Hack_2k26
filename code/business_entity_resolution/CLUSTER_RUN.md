# Running the Business Entity Resolution pipeline on a cluster

This package reproduces the whole solution end to end. Starting from the 7 challenge TSV files, it produces the two deliverables with one command:

```
output/matching_results.tsv   <- upload to the leaderboard portal
output/candidate_pairs.tsv    <- goes into the final submission zip
```

It uses no internet, external data or APIs; `pip install` is the only download.

---

## 1. What to transfer to the cluster

| Item | Size | Source on the laptop |
|---|---|---|
| **Code package** (this folder) | < 1 MB | `code/business_entity_resolution/` (or `er_code_package.zip`) |
| **Dataset**: 7 TSV files | 2.2 GB | `student_resource/dataset/train/*.tsv` and `student_resource/dataset/test/*.tsv` |
| **Validator** (optional but recommended) | 14 KB | `student_resource/utils/validate_submission.py` |

Recreate this layout on the cluster; the default paths are resolved from it:

```
<ROOT>/
├── code/business_entity_resolution/      # this package
│   ├── src/  run.sh  run.ps1  requirements.txt  README.md  CLUSTER_RUN.md
├── student_resource/
│   ├── dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   ├── dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
│   └── utils/validate_submission.py
├── work/      # created automatically: caches (~25-35 GB)
└── output/    # created automatically: final TSVs
```

A different layout also works. Point the pipeline at your paths with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATA_DIR` | folder containing `train/` and `test/` | `<ROOT>/student_resource/dataset` |
| `ER_WORK_DIR` | cache / intermediate folder (fast local disk) | `<ROOT>/work` |
| `ER_OUTPUT_DIR` | where the two TSVs are written | `<ROOT>/output` |
| `ER_N_JOBS` | CPU worker threads/processes | number of cores − 2 |
| `ER_BACKEND` | `xgb` = XGBoost on the CUDA GPU, `lgb` = LightGBM on CPU | auto (GPU if available) |
| `ER_VALIDATOR` | path to `validate_submission.py` | `<DATA_DIR>/../utils/validate_submission.py` |

---

## 2. Requirements

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| RAM | 16 GB | **32 GB+** | Measured peak is 10.5 GB (US blocking-index build) |
| Disk (work dir) | 35 GB free | 50 GB, local SSD | Caches of normalised tables, blocked pairs, features |
| CPU | 8 cores | **16–64 cores** | String similarity (rapidfuzz) and blocking are CPU-bound and multi-threaded |
| GPU | none | 1 × NVIDIA GPU, ≥ 6 GB, recent driver | XGBoost-CUDA training/inference. Without a GPU it falls back to LightGBM CPU automatically. |
| Python | **3.12** | 3.12 | Versions pinned in `requirements.txt` |
| OS | Linux or Windows | Linux | Tested on Windows 11. The code is OS-agnostic. |

---

## 3. Setup

```bash
cd <ROOT>/code/business_entity_resolution
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py --check
```

`--check` prints library versions, the chosen backend (`xgb` means the GPU is used), the resolved paths, and `data files OK (7/7)`. Fix anything it flags before starting the run.

---

## 4. Run

```bash
cd <ROOT>/code/business_entity_resolution
bash run.sh                     # full run; log also written to <WORK_DIR>/run_all.log
```

Keep it alive through a scheduler, `tmux`/`screen`, or `nohup bash run.sh &`. On SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=er-pipeline
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
cd <ROOT>/code/business_entity_resolution
source .venv/bin/activate
export ER_WORK_DIR=/local/scratch/$USER/er_work          # fast local disk if available
bash run.sh
```

### Resume after an interruption

Every step caches its output, and blocking also resumes at chunk level, so nothing is recomputed.

```bash
bash run.sh --from pairs_test          # any step name from the table below
```

### Steps and measured times (8-core laptop, RTX 4050)

| # | Step | What it does | Laptop time |
|---|---|---|---|
| 1 | `preprocess` | Parallel normalisation of names and addresses (24M records) | ~8 min |
| 2 | `split_countries` | Per-country S1 / pool tables (country is an open set) | ~5 min |
| 3 | `translit` | Learns an Indic-script → English word dictionary from train pairs | ~2 min |
| 4 | `train_ranker` | Learns the blocking ranker on 3 % of train S1 (GPU) | ~15 min |
| 5 | `pairs_train` | Blocking + cheap features, train (India, US) | ~90 min |
| 6 | `pairs_test` | Blocking + cheap features, test (France, India, US) | ~70 min |
| 7 | `stage1_fit` | GPU pruner | ~1 min |
| 8–9 | `stage1_select_*` | Top-8 per source, p ≥ 0.003 → final candidate set | ~5 min |
| 10 | `submission_stage1` | Early safety-net output (stage-1 scores only) | ~8 min |
| 11–12 | `stage2_features_*` | Rich, TF-IDF, competition and consistency features | ~30 min |
| 13 | `stage2_fit` | 2-fold GPU matcher, out-of-fold predictions | ~10 min |
| 14 | `stage2_predict_test` | Test scores | ~2 min |
| 15 | `stage3` | Re-ranking with competition + cluster-consistency features | ~15 min |
| 16 | `submission` | Tunes thresholds on OOF train, writes both TSVs | ~8 min |
| 17 | `validate` | Official validator | ~1 min |

The total is about **4.5–5 h on the laptop**. With 32+ cores, steps 1, 5, 6 and 11–12 scale roughly linearly, so expect about **1.5–2.5 h**.

---

## 5. How to read the results

The log prints the validation score of the final model (out-of-fold on 1.99M held-out train S1):

```
[decide] OOF macro-F0.5 = 0.98xxx with {'margin': ..., 'taus': [...]}
   India: 0.97xxx   US: 0.98xxx   singletons: ...
[submit] test S1=1,732,544 mean candidates=... mean matches=... empty rows=...%
PASS — no blocking issues found. Safe to submit.
```

**Bring back from the cluster:**
- `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- `<WORK_DIR>/decision.json`, which holds the chosen thresholds and the OOF score
- `<WORK_DIR>/run_all.log`

The intermediate caches in `work/` are not needed afterwards.

**Sanity values from the laptop run (v1 / v1.5):**
- Test: 1,732,544 rows, about 7 candidates and about 3.3 matches per S1, 5.6–5.8 % empty rows in every country (US, India, France).
- OOF F0.5: 0.9792 (v1), 0.9808 (v1.5). This package (v2, learned blocking ranker) is expected to score about 0.984.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `--check` says `backend lgb` but you expected the GPU | `nvidia-smi` must work; install the NVIDIA driver; `pip install xgboost==3.4.1`. Force it with `ER_BACKEND=xgb` to see the error. |
| Killed / out of memory during `pairs_*` or `stage2_features_*` | Give the job more RAM (32 GB+). Lower `ER_N_JOBS` if many cores multiply the worker memory. |
| `MISSING DATA FILES` | Set `ER_DATA_DIR` to the folder that contains `train/` and `test/`. |
| Validator says `FAIL` | Should not happen: files are UTF-8, LF, tab-separated, and matches are checked as a subset of candidates in code. Send the validator output back. |
| Want a clean re-run | Delete `<WORK_DIR>` (or point `ER_WORK_DIR` to a new folder). `ER_FORCE=1` forces blocking to recompute. |

---

## 7. Context: the problem and what this pipeline does

**Task.** For each of 1.73M Source-1 businesses, find every Source-2/3 record (9.97M) of the same business. The metric is macro F0.5 per S1, so precision is weighted 2×, and singletons score 1 only when the prediction is empty. The final ranking also rewards a small candidate set per S1. France (15 % of test) never appears in train.

**Measured data facts that drove the design** (full analysis in `docs/PROBLEM_STATEMENT.md`):
- Country is a perfect hard block.
- Each S2/S3 record belongs to at most one S1.
- 38 % of S1 names are shared by different businesses, so address decides.
- 13–23 % of Indian S2/S3 names are in native scripts.
- Postal codes are almost absent.
- Noise includes OCR swaps, legal-form moves, domains, and `dba`/`aka` aliases with made-up names.

**Pipeline:**
1. **Normalisation.** ASCII folding, alias splitting, OCR repair, legal-form canonicalisation (US, India, France), address abbreviations, state ↔ code, house-number cleanup, a phonetic skeleton, and a transliteration dictionary learned from train pairs.
2. **Blocking.** 8 complementary keys per country: exact name, no-space name, rare token, token pair, skeleton pair, (house number, address word), address-word pair, (house number, name token). A cheap evidence pre-cap is followed by a **learned GBDT ranker** that keeps the top 15 per (S1, source). Train recall is 0.989 at about 30 pairs per S1.
3. **Stage 1.** A GBDT pruner keeps the top 8 per source with p ≥ 0.003. This is the `candidate_pairs.tsv` set: about 6–7 per S1, and it loses less than 0.3 % of reachable true pairs.
4. **Stage 2.** A GBDT matcher on about 70 country-agnostic features: string similarities, IDF/TF-IDF, frequency (ambiguity), competition and consistency features. It is trained 2-fold by S1 on the GPU.
5. **Stage 3.** A GBDT re-ranker with competition and cluster-consistency features recomputed from the stage-2 scores.
6. **Decision.** Each pool record goes to its best S1. Rank-specific thresholds (1st / 2nd / 3rd+ match) are tuned for exact macro F0.5 on out-of-fold train predictions. Caps: at most 5 S2 and 6 S3 per S1.

**Compliance.** Final models are XGBoost (Apache-2.0) or LightGBM (MIT). No pretrained models, external data or lookups are used.
