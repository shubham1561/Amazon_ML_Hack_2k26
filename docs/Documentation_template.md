# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27-Sep-2026

---

## 1. Executive Summary

A four-stage pipeline:

1. **Multi-key blocking** with similarity-ranked caps (8 complementary keys, vectorised joins).
2. A **LightGBM pruner** that produces the final candidate set.
3. A **LightGBM matcher** on about 70 country-agnostic similarity, frequency, competition and consistency features.
4. A **constrained decision layer**: each S2/S3 record goes to at most one S1, per-source caps apply, and thresholds are tuned for the exact macro-F0.5.

Key ideas:
- Evidence is frequency-aware, to handle the 38 % of S1 names shared by several businesses.
- A transliteration dictionary is mined from the training pairs for Indian scripts.
- A phonetic skeleton recovers transliterated names.
- The decision is F0.5-aware: a low bar for the first match, a high bar for extra ones.

---

## 2. Methodology

### 2.1 Problem Analysis

Full EDA is in `PROBLEM_STATEMENT.md`. Measured facts that drove the design:

- **Scale:** test has 1.73M S1 records and 9.97M S2+S3 records. Train has 2.21M S1 and 10.3M S2+S3.
- **Country is a perfect hard block:** 0 cross-country true pairs. France (15 % of test S1) is absent from train, so country is treated as an open label and never used as a feature.
- **Every S2/S3 record belongs to at most one S1** (0 of 7.64M matched IDs are shared). Singletons are 5.6 %, the mean is 3.46 matches per S1, and the maxima are 5 (S2) and 6 (S3).
- **Name twins and decoys:** 38.3 % of S1 share their exact name with another S1 (e.g. "Primary Care Group" ×253). 67–74 % of S1s have a same-name S2/S3 record that is *not* their match. Only 52 % of true pairs have equal normalised names.
- **Indian scripts:** 23 % of Indian S2 names and 13 % of Indian S3 names are in native scripts (Devanagari, Kannada, Malayalam, Tamil, Telugu, Gujarati, Bengali, Gurmukhi).
- **Other noise:** OCR digit swaps (5tartups, Br0de, 6lobal), accent injection, legal-form moves (Co Mvk Leasing), junk wrappers, phone numbers and IDs in names, domain names (hariomfood.com), coined aliases ("Brixsol dba Feor Holdings", "Deltaxylofaye").
- **Addresses:** components reordered or truncated, zero-padded or suffixed house numbers, abbreviations, native-script state names, `null` tokens, 3 % empty. Postal codes are almost absent, so they are not used for blocking.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage GBDT + constrained assignment (multi-stage, as in the top Kaggle *Foursquare Location Matching* solutions).  
**Core Innovation:**
- Frequency- and competition-aware features (Fellegi-Sunter-style term-frequency ideas, used as learned features).
- A transliteration dictionary learned from the training pairs, plus phonetic skeletons.
- A decision layer that exploits the one-owner-per-record constraint and the shape of the F0.5 metric.

---

## 3. Candidate Generation (Blocking)

Everything runs per country. S1 is blocked against the pool (S2 ∪ S3), and every join is a vectorised sort/searchsorted expansion.

**Blocking keys used (union):**

| Key | Definition | Block cap |
|---|---|---|
| NAME | exact sorted normalised name (pool: name and `dba/aka` alias) | 300 |
| NOSP | exact name without spaces (`hariomfood.com` ↔ `Hariom Food`) | 300 |
| TOK | rare single name token (df ≤ 300) | 300 |
| TOK2 | pair of name tokens among each record's 4 rarest | 300 |
| SKEL | pair of phonetic-skeleton tokens (pool: Indian-script names) | 300 |
| ADDR | (house number, address word), 3 numbers × 8 rarest words | 300 |
| AW2 | pair of rare address words (addresses without numbers) | 300 |
| NUMT | (house number, name token) | 150 |

**Ranking and caps:**
1. Each matched key adds 1/log2(2 + block size) to a pair's evidence score. The top 300 pairs per S1 by evidence go on.
2. A fast similarity (name / skeleton `token_set_ratio` + address `token_set_ratio` + evidence) ranks those, and the top 40 per (S1, source) are kept.
3. A LightGBM pruner, trained on cheap features for 10 % of train S1, keeps the top K per (S1, source) with p ≥ p_min.

Output of step 3 = `candidate_pairs.tsv`, which is exactly the set the matcher scores.

- **Candidate pairs generated:** [TODO total] test pairs, [TODO] per S1 on average.
- **Blocking recall (train, held-out S1):** union [TODO], after the similarity cap [TODO], final candidate set [TODO].
- **How we protected recall:**
  - The 8 keys are complementary: name, token, transliteration and address paths each reach cases the others miss.
  - Caps are applied by similarity, not arbitrarily.
  - Every change was measured against ground truth with diagnostics of the missed pairs.

---

## 4. Matching Model

**Normalisation:**
- ASCII folding
- junk / URL / ID / phone stripping
- alias split (dba, aka, a/k/a, formerly known as, née, DBA:)
- OCR digit→letter repair
- legal-form canonicalisation (US, India and French forms) and removal from the core name
- honorifics removed
- address abbreviations (English and French), state name ↔ code, ordinal words, house-number cleanup, unit and PO-box extraction
- Indian-script names: romanised, then mapped through the learned dictionary

**Features used:**
- **Name:** token-set, token-sort, partial and plain ratios, Jaro-Winkler, Levenshtein, no-space ratio, skeleton token-set, alias token-set, filler-free token-set, char-3gram TF-IDF cosine, IDF-weighted coverage on both sides, max shared IDF, first-token match, legal-form agreement, token counts.
- **Address:** token-set, token-sort, partial and plain ratios, IDF-weighted coverage, max shared IDF, house-number overlap / first-number match / Jaccard, unit match, PO box, empty-address flag.
- **Ambiguity:** how many S1 share this name, how many pool records share it, how many S1 share the candidate's name.
- **Competition (context):** stage-1 score, rank within the S1 (overall and per source), best / second-best / gap, high-confidence counts; rank among the S1s claiming the same pool record, best competing score, margin.
- **Consistency:** name and address similarity to the S1's most confident other candidate.
- **Block evidence:** the 8 key bits, evidence score, fast similarity score.

**Model type:**
- LightGBM binary classifiers: pruner (300 trees) and matcher (900 trees, 255 leaves).
- The matcher is trained 2-fold by S1 entity, giving out-of-fold predictions on train. Test = mean of the fold models.

**Threshold selection method:**
- Exact vectorised macro-F0.5 on out-of-fold train predictions. The grid covers the ownership margin, τ_lo (first match) and τ_hi (additional matches).
- Caps: at most 5 S2 and 6 S3 per S1.

---

## 5. Results & Error Analysis

Out-of-fold (OOF) macro-F0.5 on the 1.99M held-out train S1. These exclude the 10 % used to fit stage 1, and every S1 counts, including those whose true matches blocking missed.

| Version | Blocking cap ranking | Final stage | OOF F0.5 | US | India | Singletons |
|---|---|---|---|---|---|---|
| stage 1 only | formula, 40/src | p1 + ownership + rank thresholds | 0.9465 | 0.9576 | 0.9300 | 0.899 |
| v1 | formula, 40/src | stage 3 (p2 competition) | 0.9792 | 0.9817 | 0.9753 | 0.975 |
| v1.5 | formula, 40/src | stage 3 + cluster consistency | **0.9808** | 0.9832 | 0.9773 | 0.976 |
| v2 | learned ranker, 15/src | stage 3 + consistency | [TODO] | | | |

**Loss decomposition (v1):** 1 − F = 0.0208.
- Missing candidates cost 0.0075. With a perfect matcher on our candidates, the score would be 0.9925.
- Matcher errors cost 0.0134:
  - a true candidate not selected: 0.0057
  - an empty prediction for a non-singleton: 0.0042
  - false positives on non-singletons: 0.0041
  - false positives on singletons: 0.0014

The v2 blocking ranker attacks the first term. Within the same pre-cap union, the ranker keeps 99.89 % of true pairs at 10 per source, against 97.36 % for the formula.

**Blocking funnel (train, v1):**
- Union recall: US 0.993, India 0.988.
- After the similarity-ranked cap: US 0.981, India 0.971.
- After the stage-1 pruner: US 0.980 and India 0.969, at 5.5 / 6.3 candidates per S1.
- Test: 7.3 candidates per S1 (US 6.1, India 7.0, France 11.5).

**Test sanity (v1):** matches per S1 are US 3.38, India 3.33 and France 3.22. Empty rows are 5.7–5.8 % in every country, consistent with the 5.6 % singleton rate. This suggests the country-agnostic model transfers to France.

- **Public leaderboard:** [TODO]
- **Common false positives:** same-name businesses (chains and generic names such as "Primary Care Group") whose pool record has an empty or truncated address, where only the name agrees.
- **Common false negatives:** coined aliases or domain-style names with empty addresses, Indian-script names whose romanisation is outside the learned dictionary, and heavily truncated addresses ("12, MUMBAI CITY, Maharashtra").

---

## 6. Conclusion

[TODO]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`: `preprocess.py`, `normalize.py`, `data.py`, `translit.py`, `blocking.py`, `features.py`, `stage1_pairs.py`, `stage1_model.py`, `stage2.py`, `stage3.py`, `decide.py`, `make_submission.py`, `evaluate.py`, `run_all.py`
- `README.md` and a pinned `requirements.txt`

Entry point: `python src/run_all.py`.

### B. Compliance

- No external data, APIs, geocoding or lookups. Only the provided train/test files are used.
- The transliteration dictionary and IDF statistics are computed from the provided data.
- Final models are LightGBM (MIT). No pretrained model is used. Other libraries: numpy, pandas, scipy, scikit-learn (BSD), rapidfuzz (MIT), Unidecode.
