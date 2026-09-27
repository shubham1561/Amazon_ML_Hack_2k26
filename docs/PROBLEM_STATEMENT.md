# Business Entity Resolution Challenge: Complete Problem Brief and Playbook

> **What this is:** a single reference for the 72-hour ML Challenge. It combines four things: the portal problem statement, `student_resource/README.md`, the validator (`utils/validate_submission.py`), the documentation template, and a **full exploratory pass over all 7 data files** (≈2.2 GB, ~34M rows). Every number below was **measured on the actual data** on 2026-09-26. None of it is guessed.
>
> **Clock:** the pasted portal countdown read **01 d 00 h 47 m**, and the dataset was downloaded at 23:11 IST on 26-Sep-2026, so the **deadline is ≈ 28-Sep-2026 00:00 IST** (estimate). Plan for **~24 hours of work**. Confirm the exact deadline on the portal.

---

## 0. TL;DR: the 15 facts that decide this competition

| # | Fact (measured) | Consequence |
|---|---|---|
| 1 | Test = **1,732,544 S1** vs **9,969,589 S2+S3** records. Brute force = 1.7 × 10¹³ pairs. | Blocking is mandatory, and it is **judged**: a smaller candidate set per S1 ranks higher. |
| 2 | **Country is a perfect hard block**: 0 cross-country matches in 207,808 sampled true pairs. | Partition everything by `country` (open set: `France` is test-only). |
| 3 | **Every S2/S3 record belongs to at most ONE S1** (0 of 7.64M matched IDs are shared). | Treat it as an assignment problem. When two S1s claim the same record, at least one claim is a false positive, so keep the best. |
| 4 | Singletons (S1 with no match) = **5.58 %** of train. Mean matches per S1 = **3.46** (max 11; S2 ≤ 5, S3 ≤ 6 per S1). | An empty prediction is almost always worth **0**. Never leave a probable non-singleton empty. |
| 5 | **38.3 %** of S1 records share their *exact* name with another S1 (e.g. "Primary Care Group" ×253). | Name alone cannot decide. **Address is the tie-breaker.** |
| 6 | **66.9 %** of matched S1s and **74.2 %** of singletons have a same-normalized-name S2/S3 record that is **NOT** their match (decoys). | Name-equality ≠ match. Model "how common is this name" explicitly. |
| 7 | Only **51.9 %** of true pairs have equal normalized names. **14.4 %** share **no** name token at all (Indic script, domain names, coined aliases). | Needs address-driven recall paths. |
| 8 | 26 % of S2/S3 records match nothing (distractors). The test pool per S1 is **5.75** vs **4.68** in train. | The test is likely *more* distractor-heavy. Calibrate thresholds conservatively. |
| 9 | **France = 15 % of test S1** (259,452) and **0 % of train**. | The model must be country-agnostic. No country one-hot, and handle SARL/SAS/EURL, "R."/"Rue", accents. |
| 10 | Postal codes are **nearly absent** (5-digit in ~11 % of US addresses, 6-digit PIN in ≤1.2 % of Indian addresses, ~0.5 % in French addresses). | **Don't** use ZIP/PIN as a blocking key, even though the template suggests it. |
| 11 | 23 % of Indian S2 names and 13 % of Indian S3 names are in **Indic scripts** (Devanagari, Kannada, Malayalam, Gurmukhi, Telugu …). | Transliteration is required. The training GT is a free parallel corpus for it. |
| 12 | Metric is **macro** F0.5 per S1: `F = 1.25·TP / (1.25·TP + 0.25·FN + FP)`. One FP costs 4× one FN. | Use a low threshold for the *first* match and a high one (~0.7) for extra matches (§5). |
| 13 | Perfect precision with only 1 correct match per S1 scores just **0.696**. | Recall still matters a lot. Don't over-prune. |
| 14 | pandas on Windows writes **CRLF** by default. The validator strips only `\n`, so `\r` stays glued to the last ID of each row. The validator still **PASSES** it. | Always write with `newline="\n"` / `lineterminator="\n"` (§6.4). |
| 15 | Hardware: **16 GB RAM**, 8-core Ryzen 7 7435HS, RTX 4050 **6 GB**. | Process **per country**, use int IDs and float32 sparse matrices, stream where possible. |

---

## 1. Workspace map

```
C:\Users\shubh\Downloads\Harsh ML Hackathon\
├── PROBLEM_STATEMENT.md                      ← this file
└── student_resource\
    ├── README.md                             official problem statement (same as portal text)
    ├── Documentation_template.md             methodology template → fill it and ship it in the zip (keep the name)
    ├── utils\validate_submission.py          stdlib-only format validator (details in §6.3)
    └── dataset\
        ├── train\
        │   ├── train_source1.tsv   210 MB   2,206,821 rows   (US 1,323,633 · India 883,188)
        │   ├── train_source2.tsv   489 MB   5,034,616 rows   (US 3,016,817 · India 2,017,799)
        │   ├── train_source3.tsv   504 MB   5,285,603 rows   (US 3,170,056 · India 2,115,547)
        │   └── train_ground_truth.tsv 127 MB 2,206,821 rows  (one row per train S1)
        └── test\
            ├── test_source1.tsv    175 MB   1,732,544 rows   (India 809,986 · US 663,106 · France 259,452)
            ├── test_source2.tsv    509 MB   4,887,273 rows   (India 2,312,565 · US 1,871,330 · France 703,378)
            └── test_source3.tsv    506 MB   5,082,316 rows   (India 2,405,000 · US 1,945,701 · France 731,615)
```

**Folders you must still create** (see §6.5): `output/` (both TSVs) and `code/business_entity_resolution/{src/, README.md, requirements.txt}`.

### Environment (this machine)

| Item | Value |
|---|---|
| OS / shell | Windows 11, PowerShell 5.1 (use `;` not `&&`) |
| CPU / RAM / GPU | Ryzen 7 7435HS (8C/16T) · 15.8 GB RAM · RTX 4050 Laptop 6 GB |
| Python | 3.12.10 |
| Installed | pandas **3.0.6** (Copy-on-Write, new default `str` dtype), numpy 2.5.3, scikit-learn 1.9.1, **lightgbm 4.7.0**, **rapidfuzz 3.14.6**, unidecode |
| **Not** installed | pyarrow, polars, torch, xgboost, faiss, sentence-transformers, jellyfish (install and **pin** any you adopt) |

---

## 2. The problem, stated precisely

- **Sources.** There are three independent business-record sources with **no shared identifiers**.
  - **Source 1 (S1)** is the **deduplicated reference**: exactly one row per real business.
  - **S2** and **S3** are noisy. They can contain **several records of the same business** (up to 5 S2 and 6 S3 per S1 in train).
- **Task.** For **every S1 entity in `test_source1.tsv`**, output the set of S2/S3 `entity_id`s that refer to the same real-world business. The set may be empty (a *singleton*).
- **Schema** (all 4 source-file columns are strings):

| Column | Notes |
|---|---|
| `entity_id` | `S1-`/`S2-`/`S3-` + an integer with **no leading zeros and variable length** (1–9 digits). S2 and S3 numeric parts **collide** (26,801 overlaps in train), so always keep the prefix. There is no ID overlap between train and test. |
| `business_name` | Never empty. Abbreviations, legal suffixes, typos, transliterations, aliases (see §4). |
| `business_address` | Empty in ~2.3–3.5 % of S2/S3 rows. It contains the literal token `null` in ~2–3 % of them. It is never empty in S1. |
| `country` | `US`, `India` (train); `US`, `India`, `France` (test). Treat it as an **open set of string labels**. |

- **Ground truth** (`train_ground_truth.tsv`): `source1_entity_id`, `matched_entity_ids` (comma-separated, empty for singletons). It has one row for **every** train S1, with no duplicates and only S2/S3 IDs.
- **Two outputs** are required: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (audited). See §6.
- **Metric:** macro-averaged F0.5 over all S1 entities, singletons included. See §5.
- **Rankings:** the public leaderboard uses a subset of test and the **private leaderboard uses the rest**. The private one decides. On top of that, **the candidate-set size per S1 is reviewed and a smaller set ranks higher.**

---

## 3. Data facts (measured)

### 3.1 File integrity

| Check | Result | Implication |
|---|---|---|
| Fields per line | **exactly 4 on every line of every file** | No embedded tabs. A simple `split("\t")` is safe. |
| Line endings / BOM | LF, no BOM, valid UTF-8 | Open with `encoding="utf-8"` (the Windows default cp1252 **crashes** on Devanagari). |
| `"` characters | 0–349 lines per file contain quotes | Use `quoting=csv.QUOTE_NONE`, or default pandas quoting can swallow fields. |
| NA-like strings | 49–61 names in test S2/S3 are `null`/`None`/`NA`-like. Empty addresses number in the 100k+. | Use `keep_default_na=False, na_filter=False`, or you get NaN floats. |
| Duplicate IDs | none in any file | — |
| Exact dup (name, address) in S1 | **0**, so S1 is truly deduplicated | Two S1s with the same name are **different** businesses. |
| Train/test overlap | 0 IDs, 0 identical S1 (name, address) records | Test entities are all new. Nothing can be memorized. |
| Leakage | Row order and ID numbers are uncorrelated with matches (Pearson ≈ −0.003) | No shortcut exists. Don't look for one. |

### 3.2 Ground-truth structure (train, full file)

| # matches per S1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 count | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |
| share | **5.6 %** | 5.4 % | 17.0 % | 24.1 % | 21.9 % | 14.6 % | 7.5 % | 2.9 % | 0.8 % | 0.2 % | 0.02 % | 0.002 % |

- Mean is **3.46** matches per S1 (3.67 among non-singletons). The mode is 3.
- **S2 per S1:** 0: 287,745 · 1: 789,108 · 2: 652,779 · 3: 333,957 · 4: 119,078 · 5: 24,154 (**max 5**).
- **S3 per S1:** 0: 266,276 · 1: 716,417 · 2: 668,375 · 3: 372,443 · 4: 145,116 · 5: 35,378 · 6: 2,816 (**max 6**).
- **7,638,365** matched IDs, **each owned by exactly one S1**.
- Unmatched S2/S3 (pure distractors): S2 1,340,997 (**26.6 %**), S3 1,340,857 (**25.4 %**).
- Singleton rate by country: US 5.5 %, India 5.6 % (from a 60k-S1 sample).
- True pairs split about evenly: S2 48 %, S3 52 %.

### 3.3 Name twins and decoys: the central difficulty

- **38.3 %** of train S1 records share their exact `business_name` with ≥1 other S1 in the same country. The figure is **50.6 %** after normalization (lowercase, ASCII-fold, drop legal words).
- The top repeated S1 names are all generic US medical groups: *Primary Care Group* (253), *Ear Nose & Throat Group* (251), *Pediatric Group* (222), *Womens Health Group* (220), *Physical Therapy Group* (218) … **8,131 names occur ≥ 20 times.**
- **38.3 %** of singletons have an exact-name twin in S1.
- In a 60k-S1 sample, **66.9 %** of matched S1s and **74.2 %** of singletons had ≥1 S2/S3 record with an **identical normalized name that is not their match**.
- About a third of the *unmatched* S2/S3 records have a normalized name that equals some S1 name.

> **So:** a matcher that says "same name ⇒ match" will destroy precision. Features must tell the model **how ambiguous the name is**: how many S1s share it, and how many pool records share it. They must also carry strong **address agreement** signals (house number, street, city).

### 3.4 How often simple keys cover true pairs (blocking recall; 207,808 true pairs from 60k S1)

| Key / condition (same country) | Recall of true pairs |
|---|---|
| Equal normalized name (legal words removed, sorted tokens) | 51.9 % |
| Share ≥ 1 non-legal name token | 85.6 % (US 91.6 %, **India 76.8 %**) |
| Share ≥ 1 number in address | 79.9 % |
| Share a name token **OR** an address number | **97.7 %** |
| Matched record has **empty address** | 4.5 % of true pairs |
| Matched name in Indic script vs Latin S1 | 7.2 % of true pairs |

Why 14.4 % of true pairs share **no** name token (30k-S1 sample, share of all true pairs):

| Reason | Share of all pairs | Examples |
|---|---|---|
| Indic-script transliteration | 6.3 % | `Lotus Consulting` ↔ `लोटस कंसल्टिंग`; `Dream Care Limited` ↔ `ड्रीम केयर लिमिटेड` |
| Domain / concatenated / coined single-token name | 7.4 % | `Fio Molecular Inc.` ↔ `fiomolecularcom`; `Liu Futurewave` ↔ `liufuturewave.com`; `Pacific Network` ↔ `Evopyra`; `#aditayafunds` |
| OCR-style typos and other coined names | 0.6 % | `Basalt Inc.` ↔ `8asalt Ínc.` / `Inc. Basa1t`; `Brode Gabelli LLC` ↔ `Br0de Gacil LLC`; `Keystone Automotive` ↔ `Calovioquo (ID: 18623)` |

### 3.5 Source "personalities" (pattern rates on the first 300k rows of each file)

| Pattern | S1 (US / India) | S2 (US / India) | S3 (US / India) | Test France (S1 / S2 / S3) |
|---|---|---|---|---|
| Name in ALL CAPS | 0 / 0 | **21 % / 15 %** | 3 % / 3 % | 0 / 21 % / 6 % |
| Name non-ASCII (accents or scripts) | **0 / 0** (train) | 7 % / **27 %** | 7 % / **19 %** | **16 %** / 25 % / 24 % |
| Name in Indic script | — / 0 | — / **23 %** | — / **13 %** | — |
| Junk prefix (`***`, `--`, `<<`) | 0.1 % | 1.5 % / 1.0 % | 1.4 % / 1.0 % | ~0 |
| Brackets / pipes in name | 0 / 5 % (`(India)`) | 7 % / 10 % | 7 % / 11 % | 8–10 % |
| URL / domain in name | 0 | 4.4 % / 3.4 % | 4.2 % / 3.6 % | 3.5 % |
| Legal form moved to front (`LLC X`, `Pvt. X Ltd.`) | 0 | 2.8 % / 3.0 % | 2.6 % / 3.3 % | 3.6 % |
| Repeated word (`Galaxy Galaxy`) | 0.1 % | 2.2 % / 1.7 % | 2.1 % / 1.9 % | 0.04 % |
| Alias marker (`dba`, `aka` …) | 0 | 0 | **1.3 % / 0.9 %** | 0.8 % (S3) |
| Address ALL CAPS | 0 | **90 % / 24 %** | 0 / 0 | 0 / 28 % / 0 |
| Address ends with state code (`, TX`) | **86 %** / 0 | 83 % / 0 | **4 %** (full state names) / **57 %** (`MH`, `KA`, `UP` …) | — |
| Address starts with state (reordered) | 6 % | 6 % | 0.3 % / 3 % | — |
| Unit / Suite / Apt tokens | 13 % / 3 % | **~0 %** (dropped) / 3 % | 7 % / 2 % | ~0.2 % |
| Indic script in address (state names) | 0 | — / **23.5 %** | — / **22.5 %** | — |
| Landmark (`Near`, `Opp`, `Behind`) | — / 13 % | — / 11 % | — / 9 % | — |
| Literal `null` token | 0 | 2.9 % / 2.2 % | 2.8 % / 2.1 % | — |
| Empty address | 0 | 3.6 % / 2.9 % | 3.5 % / 3.0 % | 0 / 3.2 % / 3.0 % |
| PO Box | 0 | 1.7 % | 1.5 % | — |
| 5-digit number (US ZIP?) | 11 % | 11 % | 11 % | 0.4–0.5 % |
| `bis` / `ter` house numbers | — | — | — | 4.8 % / 3.8 % / 3.8 % |

### 3.6 Train → test shift

| Aspect | Train | Test | Risk |
|---|---|---|---|
| Countries | US 60 %, India 40 % | India 46.8 %, US 38.3 %, **France 15.0 %** | Unseen country, different mix |
| S2+S3 records per S1 | **4.68** | **5.75** (India 5.82, US 5.76, France 5.53) | More distractors per S1 → more FP pressure (or more matches per S1). Watch the predicted-match rate. |
| S1 name encoding | 100 % ASCII | France: 16 % non-ASCII names, 28 % non-ASCII addresses | ASCII-fold **everything** |
| S1 name twins | 30 % duplicated names | 28 % (493,272 dup rows) | Same difficulty |

### 3.7 France (test only): what it looks like

- **Geography is tiny:** 3 regions: **Hauts-de-France** (≈88k S1: Lille, Tourcoing, Roubaix, Dunkerque, Calais), **Nouvelle-Aquitaine** (≈74k: Bordeaux, Pessac, Mérignac, La Teste-de-Buch, Lège-Cap-Ferret) and **Pays de la Loire** (≈63k: Nantes, Saint-Nazaire, Saint-Herblain, Pornic, La Baule-Escoublac). City and region blocks are **huge**, so use street and house-number keys.
- **Legal forms:** SARL 73k, SAS 52k, EURL 17k, SA 12.8k, SASU 10.7k, SCI 8.4k, EI 4.2k. Also `Cie`, `Ets`, `& Fils`, `Frères`. Add them to the legal-word list, or `SAS` becomes a fake "distinctive" token.
- **Very generic names:** `club` 22.7k, `france` 20.8k, `ecole` 15.7k, `amicale` 14.4k, `comite` 13.8k, `maison` 10.8k, `centre` 10.5k, `union`, `sportive`, `college`, `amis`, `pharmacie` … so IDF weighting is essential.
- **Address noise (same generator as US/India):** `Rue`→`R.`/`R`, `Place`→`PL`, `Avenue`→`Av`, `Saint-Herblain`→`ST.-HERBLAIN`, `(11) R Gay-luslac`, `Nº 17`, `14 B B RUE` ↔ `14B B Rue`, **region ↔ département swap** (`Pays de la Loire` ↔ `Loire-Atlantique`, `Hauts-de-France` ↔ `Nord`, `Nouvelle-Aquitaine` ↔ `Gironde`), accents injected or stripped (`UNION DÛ CONSTELLATIONS`, `Bureau Spôrt`).
- **Aliases:** `Brixtavogild aka Loic Centre SAS`, `Yumaquo a/k/a Ets Entreprises`, `Calohalo DBA: Cynophile & Frères SAS`.

---

## 4. Noise catalogue (real train examples: S1 → S2/S3 true matches)

### 4.1 Name noise

| Pattern | Real example | Handling |
|---|---|---|
| Typos (insert/delete/swap) | `Good Tech` → `Good Tthc`, `Privet`; `Gulf Towing` → `Gulf Twig`; `Yolanda Ridley` → `Yolanda Rhoiisley` | char n-gram / Jaro-Winkler / Levenshtein |
| OCR-like digit ↔ letter swaps | `Startups`→`5tartups`, `Basalt`→`8asalt`/`Basa1t`, `Brode`→`Br0de`, `Interstate`→`lnterstate` (l↔I), `Vevoify`→`Vev0ify` | Map `0→o 1→l 5→s 8→b` inside alphabetic tokens *before* comparing |
| Accent injection | `Leasing`→`Léasing`, `Limited`→`Límited`, `Inc.`→`Ínc.`, `Learning`→`Léarning` | ASCII-fold (unidecode / anyascii) |
| Indic-script transliteration | `Good Tech Private Limited` → `गुड टेक प्राइवेट लिमिटेड`; `Galaxy Projects…` → `ഗാലക്സി പ്രോജക്ട്സ്…` (Malayalam); `Sun Engineering…` → `ಸನ್ ಇಂಜಿನಿಯರಿಂಗ್…` (Kannada) | Romanize, then use a phonetic skeleton plus char n-grams. **Learn a token dictionary from train GT pairs.** |
| Legal-form variants | `Private Limited` / `Pvt Ltd` / `Private [Ltd]` / `Pvt. … Ltd.` / `(Limited)`; `LLC` / `L.L.C.`; `Inc` / `Incorporated` / `-(Inc.)`; `गुड पावर प्रा. लि.` | Canonicalize, then compare a **core name** (without legal words) plus a legal-form-agreement feature |
| Legal form moved to front / word order | `Co Mvk Léasing`, `Ltd Set Properties`, `LLC Moncada …`, `Perez Johnson Center` | Token-set / sorted-token similarity |
| Filler words appended | `… Center`, `… Services`, `… Partners`, `… (Service)`, `LLC Partners` | Drop a learned filler list, or weight by IDF |
| Tokens dropped | `Creative Bright Aspac` → `Creative Bright`; `Feor Holdings` → `Feor`; `Amravati Rural Private Limited` → `Amravati Rural` | Containment / partial-ratio features |
| Duplicated tokens | `Galaxy Galaxy Projects`, `Set Properties Properties Limited` | De-duplicate consecutive tokens |
| Honorifics | `Mr Sun Engineering`, `Dr Set Properties`, `Smt Mvk Leang Co` | Strip `mr, mrs, ms, dr, smt, shri, sri` |
| Junk wrappers / prefixes | `*** Mvk Leasing`, `-- Holloway Peak Inc`, `<< Team Ecole`, `Johnson & ((Perez))`, `Creative-Bright [Aspac]` | Strip non-alphanumerics |
| Appended IDs / phones / URLs | `DBF One Private - 6266364680`, `Olanis LLC Partners (ID: 43172)`, `… \| www.shivshakti.com` | Regex-strip `(ID: n)`, 10-digit phones, `\| www…` |
| Domain / hashtag as the whole name | `ESHORT.COM`, `interstatetradingworks.com`, `fiomolecularcom`, `#aditayafunds` | Compare the de-spaced S1 name against the de-spaced candidate (substring / char n-gram) |
| **Alias with coined name** | `Brixsol dba Feor Holdings`, `Ariavantage Co née Yolanda Ridley …`, `Iridrex formerly known as Sun Engineering Private Limited`, `Calohalo DBA: …`, `Yumaquo a/k/a …` | Split on `dba \| d/b/a \| DBA: \| aka \| a/k/a \| fka \| formerly known as \| née \| t/a`. Compare **each side** and take the max. |
| **Coined name only** (no overlap at all) | `Redyne LLC` ↔ `Deltaxylofaye`; `Gulf Towing` ↔ `Veofaye`; `Orthopedic Heritage Care` ↔ `-- Wexarc` | Address-only evidence. Risky, so only accept with an exact house number plus street plus city. |
| `&` vs `and` | `Johnson & Perez` / `Lee and Lawson` | Normalize `& → and`, or drop both |

### 4.2 Address noise

| Pattern | Real example | Handling |
|---|---|---|
| Component reordering | `OH, Columbus, 5559 Orville Avenue`; `SUFFIELD, CT, COPPERHILL ROAD` | Bag-of-components, not positional |
| Street-type abbreviations | `Road/RD/Rd`, `Drive/DR/Dr`, `Court/CT/Ct`, `Avenue/AVE/Av`, `Street/ST`, `Lane/Ln`; FR `Rue/R./R`, `Place/PL` | Canonical dictionary |
| Typos | `Domino Daive`, `Weber Rad`, `6TH AVEUE`, `ROMAIN ROLLNAD`, `KNTA KUNJ`, `SPKOANE`, `ALUQUERQUE`, `R Daid Johnston` | Fuzzy street-name similarity |
| House-number formatting | `002442` vs `2442`, `##7604`, `#1721`, `(11)`, `476.`, `Nº 17`, `Office No 0204` | Strip `#`, parentheses, leading zeros, trailing `.` |
| House-number variants | `7` vs `7A`, `9809` vs `9809-A`, `436` vs `436-438`, `7604` vs `7604 1/2`, `16/420` vs `6/420`, `313` vs `313/5`, `16410` vs `16410a` | "Compatible number" feature (prefix / range / suffix match), not only equality |
| Number words | `6th Avenue` ↔ `SIXTH AVE`, `6th Street` ↔ `SIXTH ST` | Map ordinal words to digits |
| State code ↔ full name ↔ native script | `TX`↔`Texas`; `MH`↔`Maharashtra`↔`महाराष्ट्र`; `KA`↔`ಕರ್ನಾಟಕ`; `PB`↔`ਪੰਜਾਬ`; `Kerala`↔`Keralam` | Canonical state ID (learn native-script ↔ Latin mapping from train pairs) |
| City aliases / neighbours / county | `FT WORTH`↔`Fort Worth`, `BOMBAY`↔`MUMBAI`, `Bengaluru`↔`Bangalore`, `BETHESDA CDP`, `Fairfax County`↔`ALEXANDRIA`, `Red Oak`↔`BATTLEBORO`, `Sacramento`↔`ELK GROVE`, FR `Pays de la Loire`↔`Loire-Atlantique` | **City is a soft signal, never a hard block** |
| Missing components | `WESTERLY RD, FORT WORTH, TX` (no number); `90, Halvad, Surendra Nagar, GJ` (truncated); `SHOP 5, JAIPUR, Rajasthan` | Partial-match features and "component present" flags |
| Extra components | `PO BOX 1437` added; `Unit 1407` dropped (S2 US drops units almost always); `Plot ##613` | Ignore PO Box/unit for the main match. Use them only as a bonus when both sides have them. |
| `null` / empty | `…, NAVI MUMBAI, NULL, महाराष्ट्र`; `Connecticut, 476. Copperhill Road, null, Suffield`; empty field | Remove `null` tokens. Add an explicit `addr_missing` flag. |
| Field bleed | `Incorporated, Illinois, 704 Weber Rad, # 1407` (legal word inside the address) | Robust token features |

---

## 5. The metric: macro F0.5, and how to decide what to output

### 5.1 Exact per-entity score

```
F0.5 = (1.25 · P · R) / (0.25 · P + R)  =  1.25·TP / (1.25·TP + 0.25·FN + FP)
```

| Truth \ Prediction | empty | non-empty |
|---|---|---|
| singleton (no true matches) | **1.0** | **0.0** (any prediction) |
| has matches | **0.0** | formula above (0 if TP = 0) |

The final score is the plain mean over **all** S1 entities, so an entity with 1 match weighs the same as one with 11.

**Reference: one S1 with 4 true matches.**

| Prediction | F0.5 |
|---|---|
| 4 correct | 1.000 |
| 3 correct | 0.938 |
| 4 correct + 1 wrong | 0.833 |
| 2 correct | 0.833 |
| 3 correct + 1 wrong | 0.750 |
| 1 correct | 0.625 |
| 1 correct + 1 wrong | 0.417 |
| empty | 0.000 |

README example: predict 3, where 2 are correct and the truth is 2, gives **0.714**. Use it as a unit test for your scorer.

### 5.2 Ceiling / baseline scenarios (computed on the full train GT distribution)

| Scenario | Macro F0.5 |
|---|---|
| Predict empty for everyone | 0.056 |
| Oracle: exactly **1** correct match per non-singleton, singletons empty | **0.696** |
| Oracle: 2 correct per non-singleton | 0.873 |
| Oracle: all matches except **one missed** per non-singleton | 0.871 |
| Oracle: all matches **+ 1 false positive** per non-singleton | 0.807 |
| Oracle: perfect, but **every singleton gets 1 FP** | 0.944 |

Takeaways:
1. **Recall matters.** Stopping at the top-1 match caps you at 0.70.
2. **One FP costs more than one FN**, but not overwhelmingly so (0.807 vs 0.871).
3. Singletons are only 5.6 % of the score mass, but a false merge on one zeroes that entity.

### 5.3 Decision theory you should exploit

- **First candidate vs additional candidates need different thresholds.**
  - *First candidate:* if the S1 truly has n matches and you predict only your best candidate (correct with probability p), then E[F] = p · 1.25/(1 + 0.25n) (≈ 0.71p for n = 3). Predicting empty earns only P(singleton) ≈ 0.056 a priori. So **emit the top candidate whenever p₁ is above a low threshold** (tune it; it's often around 0.1–0.3) **unless the model believes the S1 is a singleton**.
  - *Additional candidates:* the classic optimal threshold for F_β is **F\*/(1+β²) = F\*/1.25**. With F\* ≈ 0.9, **add extra candidates only when p ≳ 0.7**.
- **Best option: an expected-F0.5 optimizer per S1.** Sort candidates by calibrated p and pick the k (including k = 0) that maximizes expected F0.5 (sketch in Appendix A.4). This needs **calibrated probabilities** (isotonic regression on out-of-fold predictions).
- **Uniqueness constraint.** An S2/S3 record belongs to ≤1 S1. If two S1s claim it, give it to the higher-p S1, or drop both if the margin is tiny. That matters most for name twins.
- **Cardinality caps.** At most 5 S2 and 6 S3 per S1 (train max). Predicting more is almost surely wrong.
- **A singleton detector helps.** Useful features: max p, the gap between p₁ and p₂, the S1's name-twin count, whether the best candidate is "owned" by a better S1, and Π(1−pᵢ).

---

## 6. Deliverables and the format contract

### 6.1 `output/matching_results.tsv` (the only scored file, uploaded to the portal)

```
source1_entity_id<TAB>matched_entity_ids
S1-00001<TAB>S2-00047,S2-00193,S3-00812
S1-00002<TAB>S3-00004
S1-00003<TAB>
```

- **Exactly one row per test S1**: 1,732,544 data rows plus the header. A missing S1 means **rejection**, and an S1 not in the test set is an **error**.
- A duplicate `source1_entity_id` row means **rejection**. A duplicate ID inside one list means **rejection**.
- Only `S2-`/`S3-` IDs are allowed. `S1-` IDs (self-matches) and other prefixes mean **rejection**.
- Commas with **no spaces** and **no quoting**. An empty list means nothing after the tab (not `[]`, `nan`, `None` or `""`).
- Nonexistent (but well-prefixed) IDs are *not* rejected. They just count as false positives.
- Expected size: ~80–100 MB. Check the portal's upload size limit early.

### 6.2 `output/candidate_pairs.tsv` (not scored, but **audited and ranked**)

```
source1_entity_id<TAB>candidate_entity_ids
```

- It follows the same format rules and has one row for every test S1 (empty if blocking found nothing).
- It must be **the exact set the final model runs inference over**, which is the *last* filtering stage before the final model. It is not an earlier, looser pass, and not a post-hoc padded or trimmed list.
- **`matching_results ⊆ candidate_pairs`** must hold for every S1. The validator warns otherwise, and reviewers read that as a pipeline bug.
- Reviewers compute **recall ceiling** and **reduction ratio** from it. **A smaller average candidate count per S1 ranks higher** beyond the leaderboard.
- Size: 1.73M rows × K IDs × ~13 bytes. K = 20 gives ≈ 450 MB, which the zip compresses well.
- It must be **reproducible from the code in the zip**.

### 6.3 What the validator does and does NOT check

Run it from `student_resource/`:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

| Checks (errors, exit 1) | Does **not** check / only warns |
|---|---|
| Header equals `source1_entity_id`, `matched_entity_ids` / `candidate_entity_ids` (after `strip().lower()`) | ID existence (only with `--check-ids`, which needs a few GB of RAM) |
| TAB-separated (a comma header is flagged as CSV) | `\r` stuck to the last ID (it strips only `\n`) |
| Every test S1 present, no extra S1 rows, no duplicate rows | Matches ⊄ candidates (**warning only**) |
| No duplicate IDs within a list | Row order (irrelevant) |
| Only `S2-`/`S3-` prefixes (a space after a comma **fails** this) | A missing candidate file (warning, but it's required in the zip) |
| UTF-8 decodable | Your score |

### 6.4 Windows / pandas gotchas (verified on this machine)

1. **CRLF:** on Windows, `df.to_csv(...)` writes `\r\n` (verified with pandas 3.0.6). The validator **passes** that file, but every row's last ID becomes `S3-123\r`, and the scorer may count it as a wrong ID. **Fix:** `lineterminator="\n"`, or `open(path, "w", encoding="utf-8", newline="\n")`.
2. **NaN singletons:** reading GT or your own outputs with default pandas turns empty lists into `NaN` floats, and `str(nan).split(",")` gives `['nan']`. **Fix:** `keep_default_na=False, na_filter=False`.
3. **Spaces:** `", ".join(ids)` produces ` S2-…`, which **fails** the validator. Use `",".join(ids)`.
4. **Encoding:** the Windows default `open()` encoding is cp1252. Always pass `encoding="utf-8"`.
5. **Quoting:** read sources with `quoting=csv.QUOTE_NONE`. Never write IDs inside quotes.
6. **Int IDs:** converting `S2-123` → `123` saves memory, but S2 and S3 numbers collide. Keep `(source, int)` and rebuild the exact string with no zero-padding.
7. **pandas 3.0 Copy-on-Write:** chained assignment (`df[a][mask] = …`) silently does nothing. Use `.loc`.

### 6.5 Final submission package (every team; top teams are audited)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # identical to the leaderboard upload
│   └── candidate_pairs.tsv         # blocking output = final model's inference set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # ALL source code
│       ├── README.md               # exact end-to-end run instructions (data → blocking → matching → output)
│       └── requirements.txt        # PINNED versions
└── Documentation_template.md       # filled-in methodology (.md or .pdf; keep the name)
```

- "Anyone should be able to regenerate **both** output files from the train/test data using only what is in this folder." So: no hard-coded absolute paths, a configurable data dir, fixed seeds, and no dependence on notebooks' hidden state.
- If you use pretrained model weights, document the model name, license and parameter count, and how they are downloaded or cached.

---

## 7. Rules and compliance

| Rule | Detail |
|---|---|
| Model license and size | The **final model must be MIT or Apache-2.0 licensed and ≤ 8B parameters.** LightGBM (MIT) ✅, XGBoost (Apache-2.0) ✅. For pretrained encoders, check the model card (e.g. `paraphrase-multilingual-MiniLM-L12-v2` is Apache-2.0, `multilingual-e5-small` is MIT, LaBSE is Apache-2.0). scikit-learn is BSD-3. That's fine for utilities, but prefer LightGBM/XGBoost for the *final* classifier to be squeaky clean. |
| **No external lookup** (disqualification) | No commercial ER APIs, no business registries, **no geocoding/address APIs**, no internet data augmentation. Calling a hosted LLM API to resolve entities also counts as an external service. **Don't.** |
| Gray areas (be conservative, document) | *libpostal*: MIT code, but trained on OpenStreetMap/OpenAddresses, so it is external address data. **Avoid.** Downloaded gazetteers or city lists: **avoid**, and derive mappings from the training data instead. A tiny hand-written abbreviation table (Rd→Road, US state codes) is domain knowledge, so it's low risk, but list it in the docs. |
| Library licenses | `unidecode` is GPL-2.0+. `anyascii` is ISC, a permissive drop-in alternative. rapidfuzz is MIT. `sparse_dot_topn` is Apache-2.0, Splink is MIT, ING's EntityMatchingModel is MIT, and cleanco is MIT (all verified; see `RESEARCH_NOTES.md`). Check any other library before relying on it. |
| Using test inputs | Fitting IDF/TF-IDF vocabularies or token statistics on the **unlabeled** test sources is standard transductive practice and helps France. Mention it in the doc. |
| Fair play | Everything is reviewed. Keep the pipeline honest and explainable, especially the blocking stages that feed `candidate_pairs.tsv`. |

---

## 8. Corner-case checklist (tick these off)

### I/O and format
- [ ] Every one of the **1,732,544** test S1 IDs appears exactly once in **both** output files, including all **259,452 France** rows.
- [ ] LF line endings, UTF-8, TAB separator, exact lowercase headers, no index column.
- [ ] No spaces after commas, no quotes, no `nan`/`None`/`[]`.
- [ ] No duplicate IDs within a list (de-duplicate *after* merging S2 and S3 blocks and after alias variants).
- [ ] No `S1-` IDs in lists (never let S1 records leak into the candidate pool).
- [ ] `matching ⊆ candidates` for every row. Assert it in code before writing.
- [ ] Run the validator with `--candidate`. Run `--check-ids` once on a machine with RAM to spare (drop `--candidate` if it runs out of memory).

### Parsing and normalization
- [ ] `keep_default_na=False` so names like `NA`/`None`/`Null` survive as strings.
- [ ] Empty addresses (129k–176k per S2/S3 file) and literal `null` tokens are handled explicitly.
- [ ] ASCII-fold all text (test France S1 has accents; train S1 has none).
- [ ] Indic scripts are romanized. **Kannada, Malayalam, Telugu, Gurmukhi, Bengali, Tamil** also appear, not only Devanagari.
- [ ] Native-script **state names** (`महाराष्ट्र`, `ಕರ್ನಾಟಕ`, `ਪੰਜਾਬ`, `ఆంధ్రప్రదేశ్`) map to the same state ID as `Maharashtra`/`MH`.
- [ ] Alias markers are split: `dba`, `DBA:`, `d/b/a`, `aka`, `a/k/a`, `fka`, `formerly known as`, `née`, `t/a`.
- [ ] `(ID: 12345)`, 10-digit phones, `| www…`, `#hashtag`, `.com` are stripped.
- [ ] OCR digit↔letter substitution is fixed inside alphabetic tokens (`5tartups`, `Br0de`, `Basa1t`, `8asalt`, `lnterstate`).
- [ ] The legal-word list covers US (`LLC, L.L.C., Inc, Corp, Co, PC, P.C., PLLC, LLP, LP`), India (`Pvt, Private, Ltd, Limited, (India), OPC, LLP, प्रा. लि.`) **and France** (`SARL, SAS, SASU, EURL, SA, SCI, EI, SNC, Cie, Ets`).
- [ ] House numbers: strip `#`, `##`, `()`, `Nº`, leading zeros, trailing `.`, and handle `A`/`-A`/`1/2`/`bis`/`ter`/ranges `436-438`/`16/420`.
- [ ] Ordinal words (`SIXTH` ↔ `6th`) and street-type abbreviations are normalized in **English and French**.

### Blocking
- [ ] Partition by country **label**, with generic code and no `{US, India}` hard-coding.
- [ ] No ZIP/PIN-based blocking (codes are mostly absent).
- [ ] City is **not** a hard key (neighbouring city, county, CDP, alias, and FR region↔département swaps).
- [ ] Stop-token handling: very frequent tokens (`private`, `limited`, `llc`, `club`, `ecole`, `care`, `group` …) are excluded or down-weighted in token blocking, or blocks explode.
- [ ] There is a recall path for **Indic-script** names (address keys plus romanized phonetic keys).
- [ ] There is a recall path for **domain/concatenated** names (de-spaced n-gram key).
- [ ] There is a recall path for **coined-alias** names (address-only key: house number + street token).
- [ ] Per-S1 cap on candidates (top-K per source). Measure **pair completeness (PC)** per country before/after the cap.
- [ ] Candidate count per S1 is reported (mean/median/p95). It's a judged quantity.

### Matching and decision
- [ ] Features are country-agnostic (similarities and ratios), with **no country one-hot** and no vocab-identity features that won't exist for French.
- [ ] Ambiguity features: S1 name-twin count, pool name frequency, IDF mass of shared tokens.
- [ ] Context features: candidate rank within its S1, gap to the best, how many S1s claim this record, and whether a better S1 exists for it.
- [ ] Probabilities are calibrated on out-of-fold data before any expected-F logic.
- [ ] The uniqueness constraint (record → ≤1 S1) and caps (≤5 S2, ≤6 S3) are enforced.
- [ ] Singleton logic: empty output only when P(singleton) beats the expected F of the best non-empty option.

### Scale and memory (16 GB)
- [ ] Load per country, keep only needed columns, and use int IDs + `category` country + float32 matrices.
- [ ] Avoid materializing 10M×n-gram TF-IDF for all countries at once. Chunk the S1 side (e.g. 5k–20k rows).
- [ ] Cache normalized text and candidates to disk (parquet needs pyarrow, so pickle/npz/feather or TSV otherwise).
- [ ] Time the full test run end-to-end at least once, well before the deadline.

### Validation
- [ ] Split by **S1 entity** (group split). Never split by pair.
- [ ] Keep the **full S2/S3 pool** during validation so decoys stay present.
- [ ] Use out-of-fold predictions for all S1 so conflict resolution sees realistic competition.
- [ ] **Leave-one-country-out** (train US → eval India and vice versa) as a France proxy.
- [ ] Sanity-check test predictions against validation: % empty rows, mean predicted matches per S1, % records claimed twice, per country (France included).

---

## 9. Validation protocol (recommended)

1. **Scorer first.** Implement the exact macro F0.5 (Appendix A.3). Unit-test it with the README example (0.714) and the §5.2 table.
2. **Dev slice for fast iteration.** Take whole *geographic* slices (e.g. all train S1 in 3 US states and 3 Indian states, plus every S2/S3 record mentioning those states). Name twins and decoys then stay realistic. Iterate there in minutes.
3. **Full out-of-fold run for calibration.** Use 5-fold GroupKFold on S1 IDs over the full train set. Train LightGBM on the candidate pairs of 4 folds and predict the 5th. Then run the decision layer on all OOF probabilities and score the macro F0.5 overall and **per country**.
4. **France proxy.** Train on US only, evaluate on India (and the reverse). If the score collapses, the features are too country-specific.
5. **Report these numbers** (they go straight into the documentation):
   - Blocking: **PC** (share of true pairs present in candidates), **mean/median/p95 candidates per S1**, **RR** (see §10), per country.
   - Matching: macro F0.5, precision, recall, singleton accuracy, per country.
6. **Leaderboard discipline.** The public LB is a subset, so don't tune to it. Use it as a sanity check (format, France not broken) and trust the OOF score.

---

## 10. Recommended architecture (blueprint)

```
             ┌─────────────────────────────── per country partition (open set) ───────────────────────────────┐
 raw TSVs ─► │ 1 NORMALIZE ─► 2 BLOCK (union of keys) ─► 3 RANK & CAP (top-K/source) ─► candidate_pairs.tsv    │
             │                                                                        │                        │
             │                                                                        ▼                        │
             │                    4 PAIR FEATURES ─► 5 LightGBM (OOF-trained, calibrated) ─► 6 DECISION LAYER │
             │                                                                                   │            │
             └───────────────────────────────────────────────────────────────────────────────────┼────────────┘
                                                                                                 ▼
                                              uniqueness resolution + caps + expected-F0.5 ─► matching_results.tsv
```

**1. Normalize** (produce several views per record):
- `name_core`: ASCII-fold, lowercase, junk/URL/ID/phone strip, alias split, OCR fix, legal/filler/honorific removal, de-duplicate tokens.
- `name_nospace`: `name_core` with spaces removed, for the domain/concatenated cases.
- `name_phon`: a phonetic skeleton of the romanized name, for Indic transliterations.
- `legal_form`: the canonical class (e.g. `pvt_ltd`, `llc`, `inc`, `sarl`).
- `addr`: parsed into `house_no` (normalized plus a compatibility form), `street_core`, `city`, `state_id`, `unit`, `po_box`, and a `has_addr` flag.

**2. Blocking**: a union of cheap, high-recall keys within the country:
- (a) **rare name tokens** through an inverted index with a DF cap and IDF-scored overlap (token blocking plus meta-blocking);
- (b) **address keys** `(house_no, street_token)` and `(house_no, city)`, which catch Indic-script and coined-alias names;
- (c) **char-3-gram TF-IDF top-N** on `name_core` (chunked sparse matmul, e.g. `sparse_dot_topn`);
- (d) `name_nospace` n-gram key for domain names.

**3. Rank and cap**: score the union with a cheap similarity (or a small stage-1 GBDT) and keep the **top-K per source**. This is a legitimate extra filtering stage, and its output *is* `candidate_pairs.tsv`.
- Tune K on the PC-vs-size curve. True S2/S3 counts per S1 are ≤ 5/6, so K ≈ 8–10 per source is a sensible start, and many S1s need far fewer.

**4. Pair features** (~40–80):
- Name: token-set/sort ratio, partial ratio, Jaro-Winkler, Levenshtein, char-3-gram cosine, IDF-weighted Jaccard, first-token match, core-name exact match, max over alias variants, `nospace` containment, legal-form agreement, and script flags.
- Address: house-number exact/compatible/conflict, street similarity, city exact/fuzzy, state match, unit/PO box agreement, token Jaccard, and empty/null flags.
- Ambiguity and context: S1 name-twin count, candidate-name pool frequency, candidate rank and score gap within the S1, number of S1s whose top-K includes this record, the best competing S1 score, and source (S2/S3).

**5. Model**: LightGBM binary classifier trained on the candidate pairs (labels from GT), out-of-fold by S1 group, with isotonic calibration. Optionally add a second-stage re-scoring with **cluster features**, e.g. similarity of a candidate to the S1's other high-confidence candidates (S2↔S3 agreement), which rescues empty-address or Indic records.

**6. Decision layer:**
- Resolve record ownership: each S2/S3 goes to at most one S1.
- Apply the caps.
- Choose per-S1 k by expected F0.5, or use a two-threshold rule (low τ₁ for the top candidate, high τ₂ ≈ 0.7 for the rest).
- Output empty only for likely singletons.

**Scale reference:** within-country Cartesian space in test = India 3.82 × 10¹² + US 2.53 × 10¹² + France 0.37 × 10¹² ≈ **6.7 × 10¹²** pairs (1.7 × 10¹³ across countries). At 20 candidates per S1 (≈ 35M pairs), the reduction ratio is RR ≈ 0.999998. The judges will care more about **candidates per S1** and **PC** than about RR.

**Stretch (only if time allows):** fine-tune a small Apache/MIT cross-encoder (e.g. multilingual MiniLM, ~118M params) on hard pairs. torch is not installed, and 6 GB of VRAM limits the batch size, so GBDT is the safer main path under the time budget.

---

## 11. Time plan for the remaining ~24 hours

| Hours (from now) | Goal | Exit criterion |
|---|---|---|
| 0–2 | Loaders, normalizer v1, scorer, safe writer, dev slice | Scorer passes the §5 unit tests |
| 2–6 | Blocking v1 at full scale (per country), candidate stats, LightGBM v1 (~25 features), **first valid leaderboard submission** | Validator PASS, a SCORED status, PC ≥ 95 % on dev |
| 6–12 | Normalizer v2 (alias, OCR, transliteration, address parse), ambiguity/context features, decision layer (uniqueness + thresholds), OOF calibration | OOF macro F0.5 per country measured |
| 12–17 | France robustness (LOCO check), shrink candidates (rank and cap / stage-1 pruner), error analysis on FPs and FNs | Smaller K at the same PC. No France regression. |
| 17–21 | **Freeze.** A clean end-to-end rerun from raw data via `src/` entry point(s); README, pinned requirements | Both TSVs regenerated byte-identically (seeded) |
| 21–24 | Fill the documentation, build the zip, validate, and upload the final matching file. Keep a buffer. | Zip structure matches §6.5 |

> Submit something valid **early** (even a crude baseline) so a format problem never costs the final hours.

---

## 12. Final submission checklist

- [ ] `output/matching_results.tsv`: 1,732,544 rows + header, LF, UTF-8, validator PASS.
- [ ] `output/candidate_pairs.tsv`: same S1 coverage, the final-model inference set, matches ⊆ candidates.
- [ ] The same `matching_results.tsv` is uploaded on the portal and shows **SCORED**.
- [ ] `code/business_entity_resolution/src/` holds all code, runnable from a clean checkout.
- [ ] `README.md` lists the exact commands, the expected runtime, the RAM needed, and the data-path configuration.
- [ ] `requirements.txt` is pinned (`pandas==3.0.6`, `numpy==…`, `lightgbm==4.7.0`, `rapidfuzz==3.14.6`, …).
- [ ] `Documentation_template.md` is filled in (see §13) and placed at the zip root.
- [ ] The docs include a model license + parameter-count statement and an explicit "no external data/APIs used" statement.
- [ ] Zip named `<team_name>_submission.zip` with the exact folder structure.

---

## 13. How to fill `Documentation_template.md`

| Template section | What to write (with numbers from this brief plus your results) |
|---|---|
| 1. Executive summary | Blocking + GBDT + expected-F0.5 decision layer; the key idea (e.g. ambiguity-aware features plus the uniqueness constraint) |
| 2.1 Problem analysis | §3 facts: name twins 38 %, decoys 67–74 %, singletons 5.6 %, Indic scripts 13–23 %, postal codes absent, France unseen, pool ratio shift |
| 2.2 Strategy | Approach type (Blocking + Classifier + constrained assignment), core innovation |
| 3. Blocking | Keys (§10.2), DF caps, K per source, **total candidate pairs, mean/median/p95 per S1, PC per country, RR**, and how recall was protected (union of name, address and n-gram keys) |
| 4. Matching model | Feature list (§10.4), LightGBM params, OOF + isotonic calibration, threshold / expected-F method |
| 5. Results and error analysis | OOF macro F0.5 overall/per country, LOCO result, typical FPs (name twins at different addresses, generic names) and FNs (coined aliases with empty addresses, Indic names with no address) |
| 6. Conclusion | Lessons learned |
| Appendix A | `src/` structure plus the entry-point commands |

---

## Appendix A: Reference snippets

### A.1 Safe reading

```python
import csv
import pandas as pd

def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                       quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False)

gt = read_tsv("dataset/train/train_ground_truth.tsv")
gt_map = {s1: (m.split(",") if m else []) for s1, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
```

### A.2 Safe writing (LF, UTF-8, every S1, de-duplicated, subset assertion)

```python
def write_lists(path, col, s1_ids, lists):
    """s1_ids: all test S1 IDs in file order; lists: dict s1 -> iterable of S2/S3 IDs."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(lists.get(s1, ())))          # dedupe, keep order
            f.write(f"{s1}\t{','.join(ids)}\n")

assert all(set(matches.get(s, ())) <= set(cands.get(s, ())) for s in s1_ids)
write_lists("output/candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cands)
write_lists("output/matching_results.tsv", "matched_entity_ids", s1_ids, matches)
```

### A.3 Exact scorer

```python
def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    fp, fn = len(pred - true), len(true - pred)
    return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)

def macro_f05(pred: dict, truth: dict) -> float:
    return sum(f05(set(pred.get(s, ())), set(t)) for s, t in truth.items()) / len(truth)

assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.7142857) < 1e-6
```

### A.4 Expected-F0.5 choice of k per S1 (Monte-Carlo, independence approximation)

```python
import numpy as np

def choose_k(p, p_singleton, n_sims=256, rng=np.random.default_rng(0)):
    """p: calibrated match probabilities of one S1's candidates, sorted descending.
    p_singleton: calibrated P(this S1 has no match at all), e.g. from a small singleton classifier
    (features: max p, p1-p2 gap, name-twin count, ...). Returns k (0 = predict empty)."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return 0
    s = float(np.clip(p_singleton, 0.0, 0.999))
    q = np.clip(p / (1.0 - s), 0.0, 0.999)                 # P(candidate true | S1 is not a singleton)
    truth = rng.random((n_sims, p.size)) < q
    n_true = truth.sum(1)
    best_k, best = 0, s                                    # empty scores 1 only if it is a singleton
    for k in range(1, p.size + 1):
        tp = truth[:, :k].sum(1); fp = k - tp; fn = n_true - tp
        f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * fn + fp), 0.0).mean()
        if (1.0 - s) * f > best:
            best_k, best = k, (1.0 - s) * f
    return best_k

# choose_k([0.95, 0.9, 0.8, 0.4, 0.1], 0.02) -> 3 ;  choose_k([0.25, 0.05], 0.10) -> 1 ;  choose_k([0.25, 0.05], 0.60) -> 0
```

Why an explicit `p_singleton`: if you estimate it as Π(1−pᵢ) from the candidates alone, you massively overestimate singletons (the prior is only 5.6 %). You then output empty rows that score 0.

Caveat: true matches that blocking missed make `n_true` an underestimate. If validation shows PC < 100 %, add the expected missing count to `fn`.

---

## Appendix B: Raw EDA numbers (for the documentation)

| Quantity | Value |
|---|---|
| Train S1 / S2 / S3 rows | 2,206,821 / 5,034,616 / 5,285,603 |
| Test S1 / S2 / S3 rows | 1,732,544 / 4,887,273 / 5,082,316 |
| Train matched IDs (S2+S3) | 7,638,365 (74.0 % of pool) |
| Train singletons | 123,247 (5.58 %) |
| Mean matches per S1 (all / non-singleton) | 3.46 / 3.67 |
| Max matches per S1 (total / S2 / S3) | 11 / 5 / 6 |
| S1 exact-name shared with another S1 | 38.3 % (normalized: 50.6 %) |
| Singletons with an S1 exact-name twin | 47,173 (38.3 %) |
| Matched S1 with a same-normalized-name decoy in S2/S3 | 66.9 % |
| Singletons with such a decoy | 74.2 % |
| True pairs: equal normalized name / share token / share addr number / token-or-number | 51.9 % / 85.6 % / 79.9 % / 97.7 % |
| True pairs with no shared name token: Indic / domain-coined / other | 6.3 % / 7.4 % / 0.6 % (of all pairs) |
| Cross-country true pairs | 0 |
| Test pool per S1 (train) | 5.75 (4.68) |
| Test S1 name duplicates | 493,272 rows (28.5 %) |
| France S1 regions (last address component) | Hauts-de-France 87,960 · Nouvelle-Aquitaine 73,819 · Pays de la Loire 62,901 |

*Method:* the EDA scripts streamed all 7 files. Pair-level statistics come from random S1 samples of 30k–60k (seeded), and pattern rates come from the first 300k rows of each file (files are pre-shuffled).
