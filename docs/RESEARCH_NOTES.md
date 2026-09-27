# Research Notes: Prior Art for Large-Scale Business Entity Resolution

> **Purpose:** outside evidence to use alongside our own data evidence (`PROBLEM_STATEMENT.md`) before choosing an architecture. The scope is **methods only**. No business records from the dataset were searched, and nothing about this live competition was consulted (fair-play). Compiled 27-Sep-2026.

---

## 1. Closest analogues

### 1.1 Kaggle *Foursquare Location Matching* (2022): the most similar public competition

About 1.5M POI records with multilingual names and addresses, matched into "same place" groups at scale. The metric is a per-record mean (IoU), which behaves like our per-S1 macro F0.5. The top solutions converged on the **same multi-stage template**:

| Stage | What top teams did | Evidence |
|---|---|---|
| Candidate generation | A **union of several cheap retrievers**, each with a small k. The 7th-place team used 5 retrievers (spatial kNN 4, weighted spatial+embedding 12, word bag-of-words 4, char bag-of-words 8, name embedding 4), giving **32 candidates per record**. | 7th-place write-up |
| Recall ceiling | Retrieval alone gave a **max-IoU ceiling of 0.9778**. **Transitivity post-processing raised it to 0.9935**, because neighbours-of-neighbours recover records that direct retrieval missed. | 7th place |
| Stage-1 pruning | A **LightGBM with a few cheap features** shrank candidates memory-efficiently. **Top-40 per record** went to stage 2. | 2nd place (team 2:30) |
| Stage-2 matcher | LightGBM/XGBoost on string similarities (Levenshtein, Jaro-Winkler, Gestalt, LCS/ROUGE), TF-IDF/SVD embeddings, distances. The 2nd-place team added **XLM-RoBERTa cross-encoders**. The 7th-place team used no transformer at all. | 2nd, 7th place; Foursquare blog |
| Precision bias | Sample weights favouring precision (positives weighted about 0.8). | 7th place |
| Post-processing | Union-find/transitivity, pruning weak bridge edges, GNN node classification. 2nd place's key idea was to **adapt thresholds to the size of the groups being merged**. | 2nd, 7th place |
| Biggest gains (7th) | Hard-negative mining, ranking optimization of the retriever, and transitivity post-processing. | 7th place |

**Differences that matter for us:**
- We have **no coordinates**, so **house number + street + city** is our "GPS" and address parsing replaces spatial kNN.
- Our clusters are **stars around a clean S1 record**, so we don't need general graph clustering. Assignment plus S2↔S3 consistency is enough.
- We have an **unseen country (France)**, so character-level, language-agnostic features matter more than learned vocabularies.

### 1.2 ING *Entity Matching Model* (EMM): company-name matching at bank scale (MIT)

- **Candidates:** TF-IDF cosine on **word** and **character n-grams** (via `sparse_dot_topn`), plus **sorted-neighbourhood** indexing. Top-n per name.
- **Supervised stage:** string-similarity features, **rank-based features** (a candidate's rank and score gap among the alternatives), and **legal-form match** features.
- **"No match":** learned from negatives sampled out of the candidate pool.
- **Aggregation:** combines several name variants of one entity. That's analogous to our several S2/S3 records per S1.

**Takeaway:** rank and competition features plus legal-form agreement are standard in industrial company matching.

---

## 2. Blocking literature

| Finding | Source | Implication for us |
|---|---|---|
| **Top-k TF/IDF (BM25) blocking beats 8 state-of-the-art blockers**, including deep-learning ones, on recall at a fixed candidate-set size. Tokenizer and attribute choice matter. | Sparkly, VLDB 2023 | The backbone should be **IDF-weighted top-k per S1** (words plus char 3-grams). Learned dense blockers aren't worth the time. |
| Fellegi-Sunter with **term-frequency adjustment**: agreeing on a *rare* value is much stronger evidence than agreeing on a *common* one. | Splink (MIT) | The principled fix for our name twins ("Primary Care Group" ×253): **frequency-aware evidence** (log-DF of shared tokens, name counts in S1 and in the pool). |
| A laptop DuckDB can handle about 20M blocked comparisons. Splink links about 1M records in ~1 minute. | Splink docs/blog | Use DuckDB (MIT) for out-of-core blocking joins on 16 GB if pandas merges blow up. |
| Dense and LSH blockers (DeepBlocker, UniBlocker, neural LSH, BlockingPy ANN) | Several papers, 2021–2025 | Only useful for cross-script cases (Indic names). Probe that before adopting. |

---

## 3. Matching models

| Option | Evidence | Fit for us (24 h, 16 GB RAM, 6 GB GPU, ~35M pairs) |
|---|---|---|
| **GBDT on similarity features** | Used by every top Foursquare team; standard in industry (ING) | **Core.** Fast, handles 35M pairs, and LightGBM (MIT) is installed. |
| **Cross-encoder transformer** (Ditto: serialize both records and classify the pair) | Ditto (VLDB 2021): **96.5 % F1 on two company datasets (789K × 412K records)**. Injecting domain knowledge (highlighting numbers and key spans) helps. | **Too slow for all 35M pairs** on a 6 GB laptop GPU. Viable only to **re-score the uncertain band** (a few % of pairs). torch isn't installed. Gate it behind a throughput and gain probe. |
| **Fine-tuned small LLM** (≤ 8B) | Steiner/Peeters/Bizer 2024: fine-tuning makes small LLMs competitive, but inference cost is high | **Drop.** Impossible at millions of pairs in the time left. |
| **Unsupervised Fellegi-Sunter** (Splink) | Mature and scalable | We have 7.6M labelled pairs, so supervised learning wins. Reuse the **ideas** (m/u weights, TF adjustment) as features. |
| **Bi-encoder / metric learning** | Used by some Foursquare teams | **Drop** as the main matcher (training time, weak on house numbers). Maybe as a blocker for Indic records only. |

---

## 4. Multi-source consistency and assignment

| Finding | Source | Implication |
|---|---|---|
| Multi-source ER needs **source-aware clustering**: a cluster should respect source constraints. | FAMER (Saeedi, Peukert, Rahm) | Enforce **each S2/S3 record → at most one S1** (assignment or "mutual best"). |
| **Transitive consistency** between sources flags false positives: if A~B and B~C are predicted, A~C should also look right. Reported **+24 F1** on average over plain pairwise matchers in multi-source settings. | TransClean (arXiv 2025) | Use **S2↔S3 agreement inside a predicted cluster** in both directions. Demote a candidate that disagrees with the S1's confident members, and **promote** one that nearly duplicates a confident member. |
| Transitivity **raises the recall ceiling** (0.9778 → 0.9935 max-IoU in Foursquare 7th). | Foursquare 7th | Add **second-hop candidates**: for each confident S1 match, pull that record's near-duplicates in the other source. This recovers Indic-name and empty-address records that direct retrieval misses. |

---

## 5. Decision theory (per-S1 output set)

| Finding | Source | Implication |
|---|---|---|
| **Exact expected-F maximisation** (the GFM algorithm) in O(m²)–O(m³) given probabilities | Dembczyński et al., NeurIPS 2011; Waegeman et al., JMLR 2014 | With ≤ ~20 candidates per S1, the **exact** optimum is cheap. It replaces the Monte-Carlo sketch in the brief (A.4). |
| The optimal global threshold for F_β is **F\*/(1+β²)** | Lipton et al. 2014 (F1 case) | With F\* ≈ 0.99, a global threshold is ≈ 0.79 for *additional* matches. The first match per S1 needs a much lower bar (see brief §5.3). |
| Adapt thresholds to group size | Foursquare 2nd place | The threshold should depend on how many confident matches the S1 already has. GFM does this automatically. |

---

## 6. Tools (licenses verified)

| Tool | License | Use |
|---|---|---|
| `sparse_dot_topn` (ING) | **Apache-2.0** | Fast chunked sparse TF-IDF top-k (blocking) |
| ING EntityMatchingModel | MIT | Reference design, optional code reuse |
| Splink | MIT | Ideas (TF adjustment) or FS features. DuckDB backend. |
| DuckDB | MIT | Out-of-core blocking joins |
| cleanco | MIT | Legal-suffix stripping (verify its French and Indian coverage on our data) |
| IndicXlit (`ai4bharat-transliteration`) | MIT, ~11M-param model | Indic→Latin transliteration. It's a model trained on an external corpus (allowed as an MIT model ≤ 8B, but needs a torch/fairseq install). **Preferred alternative:** learn a token dictionary from our own train GT pairs (~750k Indic-script names in train S2+S3, most linked by GT to a Latin S1 name). That needs no dependency and matches the dataset's generator. |
| rapidfuzz, LightGBM | MIT | Similarities, GBDT (already installed) |

---

## 7. What the research changes in our plan

**It confirms** the backbone: **union retrieval → cheap LightGBM pruner (= `candidate_pairs.tsv`) → rich LightGBM matcher → constrained, consistency-aware decision layer.** This template won the closest analogue and is used in industry.

**New ingredients to add to the brainstorm:**
1. **BM25-style top-k** retrieval on words and char 3-grams (Sparkly), unioned with **address keys**, with k chosen from the recall curve.
2. **Frequency-aware (TF-adjusted) evidence** as first-class features, to handle name twins.
3. **Rank and competition features** (ING): the candidate's rank for this S1, and this S1's rank for the candidate.
4. **Second-hop candidate expansion** plus **S2↔S3 consistency features** (TransClean / Foursquare transitivity), for recall on Indic and empty-address records and precision on decoys.
5. **Exact GFM decision** per S1, plus **mutual-best assignment**.
6. Precision-weighted training (down-weight positives slightly).
7. An *optional*, gated **cross-encoder re-scorer** for the uncertain band only (Ditto-style serialization with numbers highlighted).

**Eliminated by evidence and constraints:** LLM fine-tuning, deep learned blockers, bi-encoder as the main matcher, unsupervised FS as the main matcher.

**Still open, to be settled by data probes (not literature):** exact K per source, the recall value of second-hop expansion, how much transliteration adds over address keys, and whether the cross-encoder's gain justifies its cost.

---

## Sources

- Foursquare Location Matching, 1st place write-up: https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution
- Foursquare Location Matching, 2nd place (team 2:30): https://www.kaggle.com/competitions/foursquare-location-matching/writeups/2-30-2nd-place-solution-colum2131-part
- Foursquare Location Matching, 7th place write-up (Future Architect blog, JP): https://future-architect.github.io/articles/20220720a/
- Foursquare blog, summary of winning approaches: https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/
- Sparkly (Paulsen, Govind, Doan, PVLDB 16(6), 2023): https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf
- Splink: https://github.com/moj-analytical-services/splink · blocking tutorial: https://moj-analytical-services.github.io/splink/demos/tutorials/03_Blocking.html
- ING EntityMatchingModel: https://github.com/ing-bank/EntityMatchingModel · sparse_dot_topn license: https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE
- Ditto (Li et al., VLDB 2021): https://arxiv.org/abs/2004.00584
- Fine-tuning LLMs for entity matching (Steiner, Peeters, Bizer 2024): https://arxiv.org/abs/2409.08185
- FAMER / multi-source ER clustering (Leipzig DB group): https://dbs.uni-leipzig.de/research/projects/famer
- TransClean (2025): https://arxiv.org/abs/2506.04006
- GFM, exact F-measure maximisation (Dembczyński et al., NeurIPS 2011): https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf
- Bayes-optimality of F-measure maximizers (Waegeman et al., JMLR 2014): https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf
- cleanco: https://github.com/psolin/cleanco
- IndicXlit: https://github.com/AI4Bharat/IndicXlit
