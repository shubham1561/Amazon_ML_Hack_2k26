# CLAUDE.md: handoff report for this repository

Read this first. It is the complete context of the project: the task, what has been built, the results so far, how to run it on this machine, what to be careful about, and what to do next. It was written on 27-Sep-2026 by the Claude session that built the pipeline on the team's laptop (Windows, 8-core Ryzen 7, 16 GB RAM, RTX 4050 6 GB). The team is moving the run to this cluster.

---

## 0. TL;DR: what the human wants from you

1. **Run the pipeline end to end on this cluster** and produce `output/matching_results.tsv` (uploaded to the leaderboard) and `output/candidate_pairs.tsv` (part of the final submission zip). One command: `bash code/business_entity_resolution/run.sh` (§5).
2. Make sure the outputs **pass the official validator** (`student_resource/utils/validate_submission.py`). `run.sh` runs it automatically as the last step.
3. If time allows, **improve the score** (backlog in §9) without breaking reproducibility.
4. **Deadline:** about **28-Sep-2026 00:00 IST** (72-hour hackathon). Check the portal for the exact time. Prioritise a valid, complete run over experiments.

The human's name/handle is `shubham1561`. The repo is public (the human's choice).

---

## 1. The challenge (Amazon ML Challenge 2026: Business Entity Resolution)

- **Sources:** three business-record sources with no shared IDs. Each record has `entity_id` (prefix `S1-`/`S2-`/`S3-`), `business_name`, `business_address` and `country`.
- **Task:** Source 1 is a deduplicated reference. For **every S1 entity in the test set**, output the set of S2/S3 records that are the same real-world business. The set may be empty (a *singleton*).
- **Metric:** macro-averaged **F0.5** per S1: `F = 1.25·TP / (1.25·TP + 0.25·FN + FP)`, averaged over all S1.
  - An empty prediction on a true singleton scores 1.0. Any prediction on a singleton scores 0.
  - An empty prediction on an S1 that has matches scores 0.
  - Precision is weighted 2× relative to recall.
- **Leaderboard** (snapshot 26-Sep 22:49 IST, 5,647 teams): #1 **0.9908**, #10 ≈ 0.990, #30 ≈ 0.988. Public LB uses a subset of test; the **private LB decides**.
- **Beyond the LB:** a **smaller candidate set per S1** (`candidate_pairs.tsv`) ranks higher in the final evaluation. That file must be exactly the set the final model scores, and `matching_results ⊆ candidate_pairs`.
- **Rules:**
  - No external data, APIs, geocoding or lookups (disqualification).
  - The final model must be MIT/Apache-2.0 licensed and ≤ 8B parameters. We use XGBoost (Apache-2.0) / LightGBM (MIT) and **no pretrained models**.
  - Do not modify `student_resource/` (organiser-provided, kept byte-identical).
- **Output format:**
  - TAB-separated, UTF-8, **LF** line endings.
  - Headers `source1_entity_id<TAB>matched_entity_ids` / `...<TAB>candidate_entity_ids`.
  - **Exactly one row per test S1** (1,732,544), in any order.
  - IDs comma-separated with **no spaces**; an empty list means nothing after the tab.
  - No duplicate IDs, and only S2/S3 IDs that exist.
- **Final submission zip:** `<team>_submission.zip` = `output/{matching_results,candidate_pairs}.tsv` + `code/business_entity_resolution/{src/,README.md,requirements.txt}` + a filled `Documentation_template.md`. See §8.

Full EDA and problem brief: `docs/PROBLEM_STATEMENT.md`. Literature review: `docs/RESEARCH_NOTES.md`.

---

## 2. Data facts that drove every design decision (measured)

| Fact | Value | Consequence |
|---|---|---|
| Test size | 1,732,544 S1 vs 9,969,589 S2+S3 (train: 2,206,821 vs 10,320,219) | Blocking is mandatory |
| Countries | train US / India; **test adds France (15 %, unseen)** | Country is an open label: never hard-coded, never a feature |
| Cross-country true pairs | **0** | Country is a perfect hard block (work per country) |
| An S2/S3 record belongs to ≤ 1 S1 | 100 % of 7.64M matched IDs | "Ownership": each pool record goes to its best S1 |
| Matches per S1 | mean 3.46, max 11 (S2 ≤ 5, S3 ≤ 6); **5.6 % singletons** | Caps; empty output only for likely singletons |
| Name twins | 38 % of S1 share their exact name with another S1 ("Primary Care Group" ×253) | Address decides; frequency-aware features |
| Decoys | 67–74 % of S1 have a same-name S2/S3 record that is *not* their match | Precision needs address agreement |
| Indian native scripts | 23 % of Indian S2 names, 13 % of S3 (Devanagari, Kannada, Malayalam, Tamil, Telugu, Bengali, Gujarati, Gurmukhi) | Transliteration: learned dictionary + phonetic skeleton |
| Indian name vocabulary | only ~740 distinct romanised words (98.5 % of pairs align word-for-word) | The learned dictionary is essentially complete |
| Postal codes | 5-digit in ~11 % of US addresses, PIN in ≤ 1.2 % of Indian ones | Not used for blocking |
| Noise | OCR digit swaps, accent injection, legal-form moves/variants, junk wrappers, phone numbers/IDs in names, domains (`hariomfood.com`), coined aliases (`Brixsol dba Feor Holdings`), reordered/truncated addresses, `null` tokens, 3 % empty addresses | Normalisation (§3.1) |
| Leakage | none (row order and ID numbers are random; no train/test overlap) | nothing to exploit |

---

## 3. What we built: the pipeline

Code lives in `code/business_entity_resolution/src/`. `run_all.py` runs every step in its own process, in order, with caching.

### 3.1 Normalisation: `normalize.py`, `preprocess.py`, `data.py`, `translit.py`

**Names:**
- unidecode ASCII folding
- strip URLs / TLDs / `(ID: n)` / phone numbers / `#`
- `&`→`and`
- alias split on `dba|aka|a/k/a|fka|formerly known as|née|t/a|DBA:` (the right side is kept as `alias`)
- OCR repair inside tokens (`0→o 1→l 5→s 8→b 3→e 7→t 4→a 6→g 9→g`)
- honorifics dropped
- legal forms (US, India, France: `llc inc corp co ltd pvt llp pc pllc sarl sas sasu eurl sa sci ei ets …`) moved out of `core` into a canonical `legal` signature
- consecutive duplicate tokens collapsed

**Addresses:**
- `null` removed
- PO box and unit extracted
- ordinals → digits
- US / Indian state names → codes
- street-type abbreviations (English and French): `st/rd/ave/dr/ct/ln/blvd/pl/rue/allee/…`
- city aliases (`bombay→mumbai`, `bangalore→bengaluru`, …)
- house numbers de-zero-padded and suffix-split
- `nums` = ordered distinct numbers

**Derived per record:**
- `skel`: phonetic consonant skeleton with rules for soft c/g, b/v/w, initial vowel, nasals, so that `गुड टेक → gudd ttek → gd tk = good tech`
- `sorted_core`, `nospace`

**Translit** (`translit.py`): a word dictionary *learned from train GT pairs* (romanised Indian word → Latin word, kept only if seen in ≥ 3 distinct businesses), applied to Indian-script records at load time.

### 3.2 Blocking: `blocking.py`, runner `stage1_pairs.py`

Per country, S1 is blocked against the pool (S2 ∪ S3). The union of 8 keys uses vectorised sort/searchsorted joins with block-size caps:

| Key | Definition |
|---|---|
| NAME | exact sorted core name (pool: name and alias) |
| NOSP | exact no-space name |
| TOK | rare single name token (df ≤ 300) |
| TOK2 | pair of tokens among the 4 rarest |
| SKEL | pair of skeleton tokens (pool: Indian-script names) |
| ADDR | (house number, address word): 3 numbers × 8 rarest words |
| AW2 | pair of rare address words |
| NUMT | (house number, name token) |

**Pre-cap:** keep the top `pre_cap = 450` per S1 by a cheap evidence score (Σ 1/log2(2 + block size)).

**Ranker cap:** fast similarities (name / skeleton `token_set_ratio`, address `token_set_ratio`, no-space ratio, evidence, key bits) are scored by a **learned GBDT ranker** (`train_ranker.py`, saved as `work/ranker.pkl` and loaded automatically by `CountryBlocker`). The top `keep_per_src = 15` per (S1, source) are kept.

**Cheap features** (`features.cheap_features`) are computed for the kept pairs and streamed to `work/pairs/<split>_<country>/cNNNN.pkl`. Chunks are atomic and resumable.

### 3.3 Stage 1: pruner, `stage1_model.py` → the candidate set

- GBDT on cheap features, trained on a subsample of the **10 % of train S1 with `i1 % 10 == 0`** (the "fit rows"). These S1 are **excluded from all validation**; every other train S1 is "honest".
- Candidate rule: keep the **top 8 per (S1, source) with p1 ≥ 0.003**.
- The result is `candidate_pairs.tsv`. It loses < 0.3 % of reachable true pairs.

### 3.4 Stage 2: matcher, `stage2.py`

**Features** (≈ 75, all country-agnostic):
- cheap features, plus Jaro-Winkler, partial, Levenshtein, filler-free, char-3gram TF-IDF cosine, IDF coverage of names and addresses
- legal-form agreement, unit match, PO box, house-number Jaccard
- ambiguity: `s1_name_freq`, `p_name_freq`, `p_name_in_s1`
- **competition** from p1: rank inside the S1 and per source, best / second / gap, claims on the same pool record, margin
- **anchor consistency:** similarity to the S1's top candidate

**Training:** 2-fold by S1 (`fold2 = (i1 // 10) % 2`) gives out-of-fold (OOF) `p2` on train. Test p2 = mean of the fold models. XGBoost on CUDA, 1,200 rounds, 255 leaves, lr 0.04, up to 7M rows per fold.

### 3.5 Stage 3: re-ranker, `stage3.py`

- Keeps the stage-2 features and p2.
- Adds competition features recomputed from **p2** (prefix `q_`).
- Adds **cluster consistency** (`c_*`): best / mean name and address similarity to the S1's *other confident* candidates (p2 ≥ 0.5), plus similarity to the top candidate.
- 2-fold by `(i1 // 20) % 2` gives OOF `p3`. Test p3 = mean of the folds.

### 3.6 Decision: `decide.py`, `make_submission.py`

- **Ownership:** each pool record keeps only its best S1 claim (margin tuned).
- **Rank-specific thresholds** `taus = [τ1, τ2, τ3+]` on p3: the r-th best candidate of an S1 is kept if p ≥ τ_r. Caps: ≤ 5 S2 and ≤ 6 S3.
- Tuned by grid + coordinate ascent on **exact vectorised macro-F0.5** over honest train S1 (1.99M), counting S1 with no candidates too.
- Writes both TSVs (LF, UTF-8, every test S1 in file order) and asserts `matches ⊆ candidates`. The chosen params and OOF score are saved in `work/decision.json`.

### 3.7 Models backend: `models.py`

- **XGBoost on CUDA** if available (`ER_BACKEND=xgb`), else **LightGBM CPU** (`lgb`).
- Parameters are written LightGBM-style and translated.
- XGBoost prints a harmless "Falling back to prediction using DMatrix … mismatched devices" warning. Ignore it.

---

## 4. Results so far

OOF macro-F0.5 is measured on 1.99M honest train S1. The laptop runs are summarised below.

| Version | Blocking cap | Final scorer | Candidates / S1 (test) | **OOF F0.5** | US | India | Singletons |
|---|---|---|---|---|---|---|---|
| stage 1 only (v1) | formula, 40 / src | p1 | 7.32 | 0.9465 | 0.9576 | 0.9300 | 0.899 |
| v1 | formula, 40 / src | stage 3 (p2 competition) | 7.32 | 0.9792 | 0.9817 | 0.9753 | 0.975 |
| **v1.5** | formula, 40 / src | stage 3 + cluster consistency | 7.32 | **0.9808** | 0.9832 | 0.9773 | 0.976 |
| stage 1 only (v2) | **learned ranker, 15 / src** | p1 | 8.41 | 0.9472 | 0.9579 | 0.9311 | 0.897 |
| **v2 (this repo's code)** | learned ranker, 15 / src | stage 3 + consistency | ~8.4 | **running on laptop**; expected ≈ 0.984 | | | |

**Blocking funnel (train, share of all true pairs kept):**

| Stage | v1 US | v1 India | v2 US | v2 India |
|---|---|---|---|---|
| Union of 8 keys | 0.9933 | 0.9877 | 0.9933 | 0.9877 |
| After the similarity-ranked cap | 0.9809 | 0.9705 | **0.9923** | **0.9842** |
| After the stage-1 pruner (final candidates) | 0.980 | 0.969 | **0.991** | **0.983** |

**v1 loss decomposition** (1 − 0.979 = 0.021):
- blocking / pruning misses: 0.0075. With a perfect matcher on v1 candidates the score would be 0.9925.
- matcher: 0.0134, split into
  - a true candidate not selected: 0.0057
  - an empty prediction for a non-singleton: 0.0042
  - false positives on non-singletons: 0.0041
  - false positives on singletons: 0.0014

v2 attacks the blocking term.

**Test sanity (v1):** about 3.3 matches per S1 in every country (US 3.38, India 3.33, France 3.22). Empty rows are 5.6–5.8 % everywhere, consistent with the 5.6 % singleton rate. France transfers.

**Submissions produced on the laptop** (not in this repo; outputs are git-ignored):
- `work/sub_v15/`: v1.5, OOF 0.9808, validator PASS. This is the best completed one.
- `output_v2/`: v2 stage-1-only output, which the final v2 files will overwrite (ETA ~16:45 IST 27-Sep).

Public LB scores had not been reported back when this was written.

---

## 5. How to run on this cluster

```bash
# 0) data: the organiser folder must sit at the repo root (it is git-ignored, never commit it)
#    <repo>/student_resource/dataset/train/{train_source1,2,3,train_ground_truth}.tsv
#    <repo>/student_resource/dataset/test/{test_source1,2,3}.tsv
#    <repo>/student_resource/utils/validate_submission.py
cd code/business_entity_resolution
python3.12 -m venv .venv && source .venv/bin/activate      # or: conda create -n er python=3.12
pip install -r requirements.txt
python src/run_all.py --check        # MUST show: data files OK (7/7); backend xgb if a GPU is present
nohup bash run.sh > run.out 2>&1 &   # or submit via SLURM (see CLUSTER_RUN.md)
tail -f run.out
```

- **Resume** after any interruption: `bash run.sh --from <step>`, using the step named in the last `=== <step>` log line. Blocking also resumes per chunk.
- **Steps, in order:** `preprocess, split_countries, translit, train_ranker, pairs_train, pairs_test, stage1_fit, stage1_select_train, stage1_select_test, submission_stage1, stage2_features_train, stage2_features_test, stage2_fit, stage2_predict_test, stage3, submission, validate`.
- **Environment variables:**
  - `ER_DATA_DIR`, `ER_WORK_DIR` (fast local disk, 35–50 GB), `ER_OUTPUT_DIR`
  - `ER_N_JOBS` (default cores − 2)
  - `ER_BACKEND=xgb|lgb`, `ER_VALIDATOR`, `ER_FORCE=1` (recompute blocking)
  - `ER_K_PER_SRC`, `ER_P_MIN` (stage-1 candidate rule), `ER_MAX_TRAIN_ROWS`
- **Resources measured on the laptop:**
  - Peak RAM 10.5 GB (US blocking-index build). Stage 2 / 3 about 10 GB. Use 32 GB+.
  - The laptop total was about 5 h. String similarity and blocking are CPU-bound (rapidfuzz, multi-threaded), so 32+ cores should cut this a lot. The GPU is used for all GBDT training and inference.
- **Done looks like:**
  ```
  [decide] OOF macro-F0.5 = 0.98x...   (with per-country and singleton breakdown)
  [submit] test S1=1,732,544 mean candidates=~8 mean matches=~3.3 empty rows=~5.7%
  PASS — no blocking issues found. Safe to submit.
  === ALL DONE
  ```
- **Hand back to the human:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `<work>/decision.json`, and the log.

`code/business_entity_resolution/CLUSTER_RUN.md` has the SLURM template, a troubleshooting table and per-step timings.

---

## 6. Gotchas learned the hard way

- **Long runs die with the session.** On the laptop, runs were killed twice by sleep / app exit. Always detach (`nohup`, `tmux`, SLURM). Everything is resumable, so never restart from scratch without a reason.
- **Memory.** Keep steps sequential; two heavy steps in parallel on 16 GB caused a `BrokenProcessPool` (OOM). On a big node you *could* parallelise countries, but the code runs them sequentially by design.
- **Caches and staleness:**
  - `work/` holds the normalised caches (`norm_*.pkl`), country tables (`country/*.pkl`, whose derived columns are cached with `dver = 2`), `translit.pkl`, `ranker.pkl`, blocked pairs, candidates (`cands/`), features (`feats/`), `oof_train.pkl`, `pred_test.pkl` and `decision.json`.
  - If you change normalisation or blocking, delete the affected cache or use `ER_FORCE=1`. Otherwise stale results are silently reused.
  - `ranker.pkl` is **auto-loaded** by every `CountryBlocker` (pass `{"no_ranker": 1}` to disable). Changing the ranker features requires retraining it and re-running blocking.
- **OOF alignment:** `stage2.fit` / `stage3` assume `oof_train.pkl` / `pred_test.pkl` rows are in the same order as the per-country feature files, concatenated in `split_by_country` order. `stage3.build` asserts this. Don't reorder those frames.
- **Honest validation:** always exclude stage-1 fit rows (`i1 % 10 == 0`) from any score. `make_submission.eval_frame` does this. `i1` / `ip` are *per-country local* row indices.
- **Output hygiene** (already handled in `make_submission.write_lists`): LF endings (pandas on Windows writes CRLF by default), no spaces after commas, every test S1 present, `keep_default_na=False` when reading TSVs (empty lists become NaN otherwise), `quoting=3`.
- **pandas 3.0:** Copy-on-Write is on, so chained assignment silently does nothing. Use `.loc`.
- **France:** there is no ground truth for it. Watch its per-country candidate and match counts in the logs: test candidates per S1 run higher for France (≈ 12.7 vs 6.8–8.4) because the pruner is less certain there, but matches per S1 and empty-row rates were in line with the other countries.

---

## 7. Guardrails

- Never modify, commit or redistribute `student_resource/` (organiser data; the files are 175–509 MB and git-ignored).
- No external data, web lookups, geocoding, pretrained models or hosted LLM calls inside the pipeline. The final model must stay XGBoost / LightGBM (Apache / MIT).
- Keep `candidate_pairs.tsv` = exactly what the final model scores. Keep it small (currently ~8 per S1).
- Validate every output with the official validator before handing it back. Keep the previous best output when experimenting (write experiments to a different `ER_OUTPUT_DIR` / `ER_WORK_DIR`).
- Commit code changes with clear messages. Never commit caches, outputs, `*.pkl` or `*.tsv` (the `.gitignore` covers this).

---

## 8. Final submission package

Required zip structure (`<team_name>_submission.zip`):

```
output/matching_results.tsv
output/candidate_pairs.tsv
code/business_entity_resolution/src/*.py
code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt
Documentation_template.md          # filled-in copy is docs/Documentation_template.md: update section 5 with the final numbers
```

- Windows: `make_package.ps1 -Team <name> -OutputDir output`.
- Linux: `bash make_package.sh <team_name> [output_dir]`.

---

## 9. Improvement backlog (highest expected value first)

1. **Finish v2 and report its OOF score.** If the cluster run completes first, that output is the new best. Compare against v1.5 = 0.9808.
2. **Matcher errors (≈ 0.013 of loss):**
   - Train stage 2 / 3 on *all* train candidates, with more rounds and seed ensembling. The GPU makes this cheap.
   - Add S2↔S3 *pairwise* consistency (does candidate r agree with another S3/S2 candidate already matched to the same S1?).
   - Try exact expected-F0.5 set selection per S1 (GFM, Dembczyński 2011) with a separate singleton classifier instead of rank thresholds.
   - Try per-country thresholds for India vs US (France unseen: fall back to global).
3. **Blocking ceiling:** the union recall is 0.988 for India and 0.993 for US.
   - The remaining misses are Indian-script names with truncated or empty addresses, and coined names with empty addresses.
   - Possible: a second-hop expansion (add near-duplicates of confident matches from the other source), or a larger `pre_cap`.
4. **Optional, gated:** a small multilingual cross-encoder (Apache / MIT, e.g. MiniLM) re-scoring only the uncertain band (0.2 < p3 < 0.8). It needs `torch` and would have to be measured on OOF first. Only worth it if 1–3 are done and there is time.
5. **Keep an eye on candidate-set size:** raising recall must not blow up candidates per S1. The judges rank smaller sets higher.

---

## 10. File map

```
README.md                                  repo quick start
CLAUDE.md                                  this report
make_package.ps1 / make_package.sh         build the final submission zip
docs/PROBLEM_STATEMENT.md                  full problem brief + measured EDA + metric analysis
docs/RESEARCH_NOTES.md                     prior art (Foursquare Kaggle, Sparkly, Splink, Ditto, GFM, ...)
docs/Documentation_template.md             methodology write-up (fill final numbers)
code/business_entity_resolution/
  run.sh / run.ps1                         one-command run (Linux / Windows)
  CLUSTER_RUN.md                           cluster guide (SLURM, timings, troubleshooting)
  requirements.txt                         pinned deps (Python 3.12)
  src/config.py        paths + env overrides
  src/normalize.py     name/address normalisation, skeleton
  src/preprocess.py    parallel normalisation → work/norm_*.pkl
  src/data.py          per-country tables, derived columns, translit, ID caches
  src/translit.py      learned Indian-script → Latin dictionary
  src/blocking.py      8 keys, joins, pre-cap, learned ranker cap
  src/train_ranker.py  ranker training (3 % of train S1, uncapped union)
  src/features.py      cheap + rich pair features
  src/stage1_pairs.py  blocking + cheap features runner (chunked, resumable)
  src/stage1_model.py  pruner fit / sweep / select
  src/stage2.py        rich + context + anchor features; 2-fold matcher
  src/stage3.py        re-ranker (p2 competition + cluster consistency)
  src/models.py        XGBoost-CUDA / LightGBM backend, ranking utils
  src/decide.py        ownership, thresholds, vectorised exact macro-F0.5
  src/evaluate.py      reference scorer
  src/make_submission.py  tune decision on OOF, write both TSVs
  src/run_all.py       orchestration, --check, validation
```

**Glossary:**
- **S1:** the reference record. **Pool:** S2 ∪ S3 of the same country.
- **i1 / ip:** local row indices within a country table.
- **Fit rows:** train S1 with `i1 % 10 == 0`, used to fit stage 1. **Honest S1:** all other train S1.
- **OOF:** out-of-fold prediction.
- **p1 / p2 / p3:** stage-1 / 2 / 3 probabilities.
- **Ownership:** the one-S1-per-record constraint.
