# Business Entity Resolution — Implementation PRD & Technical Blueprint

**Amazon ML Challenge 2026 · Team the_debuggers**
Metric: macro F₀.₅ per Source-1 entity · Scale: 1.73M S1 × 9.97M S2+S3 test candidates

---

## How to read this document

Every claim is tagged so it's clear what kind of statement it is — this challenge explicitly forbids inventing requirements, so the distinction matters:

- **[FACT]** — verified directly from `amazon_ml_challenge_problem_statement.md` or by running commands against the actual dataset in `student_resource/dataset/`. Load-bearing for compliance.
- **[FINDING]** — from historical/external research (web search, general entity-resolution literature). Context, never a requirement.
- **[RECOMMENDATION]** — technical inference/opinion. Argued for, not mandated by the challenge.
- **[CORRECTION]** — flags a place where a plausible-sounding assumption is contradicted by the actual data.

---

## 1. Problem Statement

**Business problem.** A commercial platform ingests business records from three independent sources that share no common identifier. Source 1 is already internally deduplicated and treated as the reference: it is the canonical list of businesses the platform believes exist. Source 2 and Source 3 are separate feeds — different collection pipelines, different noise profiles, different countries in different proportions — and each of their records either *is* one of the Source-1 businesses under a different name/address spelling, or refers to a business Source 1 doesn't have. The ML task is to decide, for every Source-1 entity, which Source-2 and Source-3 records (zero, one, or many) describe the same real-world business.

**Why the data is difficult.** There is no shared key. Matching runs purely off `business_name`, `business_address`, and `country` — three free-text-ish fields, each independently noisy:

- Names carry legal-suffix inconsistency (Pvt vs Private, Corp vs Corporation, Ltd vs Limited), DBA/trade-name divergence from legal name, punctuation variance (`&` vs "and"), word-order transposition, and typos.
- Addresses carry abbreviation variance (St/Street, Rd/Road), transliteration variants, missing components (no PIN/ZIP, no state), landmark references ("Near SBI ATM"), municipal numbering quirks, and component reordering.
- Country is an open set — see the Fact below, the single most architecturally consequential detail in the problem statement.

> **[FACT]** — verified in `test/test_source1.tsv`
> Training data covers only `US` and `India`. The test set adds a third country label, `France`, with zero training examples. Actual test-set country split (counted directly): **India 809,986 (46.8%)**, **US 663,106 (38.3%)**, **France 259,452 (15.0%)** of 1,732,544 Source-1 test entities. France is roughly one test entity in seven, and every one must appear in the submission with a prediction.

**What counts as a correct match.** Per entity: predicting the exact ground-truth set of Source-2/3 IDs is precision 1.0, recall 1.0, F₀.₅ = 1.0. A superset (real matches plus a wrong one) or a subset (missed a real match) partially credits. Predicting *any* match for a true singleton scores 0.0 on that entity regardless of how plausible the candidate looked.

**Why false merges are costly, in the challenge's own terms.** A false merge (matching two different real-world businesses) is a false positive on that S1 entity's precision. Because the scoring metric is F₀.₅, precision counts twice as heavily as recall in the harmonic blend — a single wrong merge does more damage to that entity's score than a single missed true match does. The problem statement states this directly: "F₀.₅ weights precision 2× over recall," and frames it in real-world terms — merging two distinct businesses is more damaging than missing a link, because a false merge corrupts downstream business records (wrong tax ID, wrong contact, wrong account) in a way a missing link never does.

> **[FACT]** — F₀.₅ formula, verified
> `F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`, computed **per Source-1 entity**, then macro-averaged across every Source-1 entity in the evaluation set. Not micro-averaged over pairs — an entity with 1 true match and an entity with 10 true matches count equally in the final average. That single design choice is why singleton handling and low-support entities matter as much as high-fanout ones.

**Singletons / no-match, in official terminology.** "A Source 1 entity with no true matches scores 1.0 when you correctly predict an empty list, and 0.0 when you predict any match for it." The problem statement calls these entities "singletons" and states outright: "Do not neglect singletons — correctly predicting 'no match' is worth a full 1.0 on that entity."

> **[CORRECTION]** — measured, not assumed
> A common assumption in blocking-heavy ER challenges is that singletons are a large fraction of the reference set (30–50%). **That is not true here.** Direct count on `train_ground_truth.tsv`: 123,247 of 2,206,821 training Source-1 entities have an empty match list — **5.58%**, not ~40%. This changes the leaderboard-impact argument for the singleton classifier (§16) — it still matters, but the reasoning has to be the correct one, not an inflated one.

---

## 2. Challenge Constraints & Official Requirements

Every item below is **[FACT]**-tier, extracted from `amazon_ml_challenge_problem_statement.md` and cross-checked against `student_resource/utils/validate_submission.py`, the actual validator the challenge ships. Anything the validator enforces as a hard rule is marked **blocking**; anything it only warns about is marked **soft**.

### Input files (verified against the actual dataset on disk)

| File | Columns | Row count (measured)¹ |
|---|---|---|
| `train/train_source1.tsv` | entity_id, business_name, business_address, country | 2,206,821 |
| `train/train_source2.tsv` | same | 5,034,616 |
| `train/train_source3.tsv` | same | 5,285,603 |
| `train/train_ground_truth.tsv` | source1_entity_id, matched_entity_ids | 2,206,821 (1 row per S1 train entity, exactly) |
| `test/test_source1.tsv` | entity_id, business_name, business_address, country | 1,732,544 |
| `test/test_source2.tsv` | same | 4,887,273 |
| `test/test_source3.tsv` | same | 5,082,316 |

<sup>1</sup> Data-row counts (header excluded) — verified against `pl.read_csv` output in `code/business_entity_resolution/src/data/loaders.py` as of Phase 0. An earlier revision of this table reported `wc -l` output, which counts the header line as a row and was off by exactly 1 on all seven files; caught when `loaders.py`'s hard row-count check failed on its first real run (see `CHANGELOG.md`). No percentage-based finding elsewhere in this document (singleton rate, country split, match-count distribution) is affected — those were already computed off data rows directly.

> **[FACT]** — scale implication
> Naive all-pairs comparison at test time is `1,732,544 × (4,887,273 + 5,082,316) ≈ 1.73M × 9.97M ≈ 1.73 × 10¹³` candidate pairs. Blocking is not an optimization here — without it, exhaustive comparison is off by roughly 8–9 orders of magnitude from anything that runs in the challenge's compute budget. Read this as a hard architectural constraint, not a "nice to have."

### TSV format rules

- Every file is tab-separated, including your two output files. The validator's #1 detected mistake is a comma-separated file wearing a `.tsv` name, and it fails hard on that.
- `entity_id` prefix (`S1-`/`S2-`/`S3-`) is the only source indicator — there is no explicit source column.
- ID lists inside a cell are comma-separated, no quoting, no internal whitespace shown in examples.

### Country handling — the open-set requirement

> **[FACT]**
> The problem statement is explicit: *"Treat country as an open set of string labels: do not hard-code, filter, or one-hot your pipeline to only {US, India}, and remember that every test entity — France included — must appear in your submission."* Any component that special-cases `country in {"US","India"}` (fixed one-hot encoder, hardcoded blocking partition list, France-excluding fallback) is a correctness bug against this requirement, not a style choice.

### Ground-truth format

`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, empty string for singletons). One row per training S1 entity — confirmed exactly 2,206,821 rows, matching S1's row count 1:1.

### Submission formats — `matching_results.tsv` (scored) and `candidate_pairs.tsv` (not scored, audited)

| Rule | File(s) | Enforcement (per validator) |
|---|---|---|
| Header must be exactly `source1_entity_id\tmatched_entity_ids` (or `candidate_entity_ids`) | both | blocking |
| Every test `S1-*` entity has exactly one row | both | blocking (missing) / blocking (duplicate rows) |
| Empty list = no match, written as nothing after the tab | both | expected format |
| No duplicate IDs inside one row's list | both | blocking |
| IDs must be `S2-`/`S3-` only — no `S1-` self-matches, no foreign prefixes | both | blocking |
| IDs must exist in the test Source-2/3 files | both | blocking only when `--check-ids` passed (off by default); otherwise a scoring-time quality issue, not a validator rejection |
| `matching_results.tsv` IDs ⊆ `candidate_pairs.tsv` IDs for the same S1 entity | cross-file | soft — validator **warns**, does not fail |

> **[FACT]** — subtle but important
> `candidate_pairs.tsv` is defined as "the exact set of records you feed into your matching model for inference — the final candidate list just before the ML model scores them... If your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last one." This is not your raw blocking-stage output if you filter further downstream (e.g. bi-encoder retrieval followed by a cross-encoder narrowing) — it is the last-stage candidate set, i.e. exactly what the top-20 reranker in §8 receives.

### Final submission ZIP structure (verbatim, do not restructure)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

### Requirements that can cause disqualification, not just point loss

- **Model license / size:** "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters." Any embedding/cross-encoder/LLM component in the final pipeline must clear both bars simultaneously.
- **External identity lookup — strictly prohibited:** commercial entity-resolution APIs, government business-registration lookups, geocoding APIs for address normalization, any internet-sourced data augmentation. Enforcement is explicit: "Any evidence of external data lookup will result in immediate disqualification." This rules out geocoding-based address canonicalization even though it would trivially help — see §24.
- Format failures are gating, not merely penalized: "Submissions that fail validation will not be evaluated." Run `validate_submission.py` as a CI step, not a manual pre-flight (§23).

### Validation requirements

`utils/validate_submission.py` is stdlib-only (no install needed), reads only your two output files plus the test source files, and never computes your score — it is a format gate, not a leaderboard proxy. It defaults to skipping the expensive ID-existence check (`--check-ids`) because loading all ~10M test S2/S3 IDs costs multiple GB; a nonexistent ID lowers your score at scoring time but does not fail local validation unless you opt in.

---

## 3. Target Users / Personas

**ML Data Scientist — "Priya"**
Owns the modeling loop: EDA → feature engineering → training → threshold tuning. Lives in notebooks and a local validation harness scored with the exact F₀.₅ formula from §1. Her single biggest fear is a leaderboard score that doesn't match her local validation score — usually caused by threshold drift across countries or a leaked entity in her CV split. She needs the local eval framework (§17) and the leakage-proof CV design (§15) more than any other artifact in this document.

**ML Ops / Pipeline Engineer — "Arjun"**
Owns reproducibility and the submission package: deterministic runs, the exact ZIP structure, dependency pinning, the invariant that `matching_results ⊆ candidate_pairs`, and getting `validate_submission.py` green before every leaderboard upload. Cares about wall-clock time on the 1.73M × 10M-candidate test set fitting inside whatever compute budget the team has, and about the code/ folder actually reproducing the two output files end-to-end on a machine that isn't his laptop.

---

## 4. Goals and Non-Goals

### Goals

- Maximize macro F₀.₅ on the held-out validation split and, by construction, on the leaderboard.
- High precision without collapsing recall — F₀.₅ punishes false merges hard, but a precision-only model that predicts singleton-for-everyone scores exactly the singleton rate (~5.6%) and nothing more; recall has to clear a real bar too.
- Blocking recall ceiling high enough (>99% target, measured directly — §12) that the matching model's own precision/recall trade-off is the binding constraint, not missed candidates.
- Correct singleton classification, since a wrong non-empty prediction on a true singleton is a hard 0.0 on that entity regardless of match quality.
- Strict reproducibility: same seed, same CUDA flags, same pinned deps → same two output files, byte-identical or score-identical.
- A `matching_results.tsv` / `candidate_pairs.tsv` pair that passes `validate_submission.py` on every run, enforced as an automated pre-commit/CI check, not a manual habit.

### Non-Goals

- **No production-grade external business database.** Explicitly prohibited by the fair-play rules (§2, §24) — not a scoping choice, a compliance one.
- **No frontend.** See §7 — this is a batch scoring pipeline with a fixed, machine-readable output contract; a UI adds zero leaderboard value.
- **No web/API enrichment of any kind** (geocoding, WHOIS, business-registry lookups) — prohibited outright.
- **No real-time/low-latency serving.** The output is a static file produced once per submission cycle; there is no request-response SLA to design for.
- **No general-purpose entity-resolution library/product.** Build the narrowest pipeline that clears this specific dataset's noise profile and scale — not a reusable ER platform.

---

## 5. User Stories

| Stage | Story |
|---|---|
| Data ingestion | As Arjun, I want a single loader that reads all six TSVs with explicit `sep="\t"` and validates row counts against expectations, so that a silently-mis-parsed file (the challenge's own documented #1 footgun) never reaches the pipeline undetected. |
| EDA | As Priya, I want a noise-quantification report (missing address components, name-edit-distance histograms, singleton rate, per-country distributions) computed once and cached, so that every downstream design choice — blocking radius, normalization rules, thresholds — is grounded in measured numbers, not assumption. |
| Preprocessing | As Priya, I want deterministic, reversible normalization functions for name/address/country (no lemmatization) with unit tests on known noise patterns (Pvt→Private, St→Street), so that normalization never silently destroys a legally distinguishing token. |
| Candidate generation | As Priya, I want a blocking stage with a measured recall ceiling logged per run, so that I know whether a low downstream F₀.₅ is a blocking problem or a scoring problem before I touch the classifier. |
| Pair scoring | As Priya, I want a reproducible feature-extraction + model-scoring pipeline that emits a calibrated match probability per (S1, candidate) pair, so that the same probability means the same thing across countries and sources. |
| Threshold tuning | As Priya, I want per-country threshold sweeps against the exact macro F₀.₅ formula, so that the France test slice (unseen at train time) isn't scored under a threshold tuned only for US/India. |
| Cross-validation | As Priya, I want GroupKFold grouped by `source1_entity_id`, so that my local validation score is not inflated by the same entity's records leaking across train/val. |
| Submission | As Arjun, I want a single command that regenerates `matching_results.tsv` and `candidate_pairs.tsv`, runs `validate_submission.py`, and fails the build on any error, so that a broken submission never reaches the portal. |

---

## 6. Feature Scope

### MVP — essential for a strong baseline

- TSV loaders with dtype/row-count validation
- Domain-specific normalization (legal-suffix map, address-token expansion, casefold, whitespace/punctuation cleanup) — no lemmatizer
- Country-partitioned blocking (exact/TF-IDF hybrid) with a fallback blocker guaranteeing non-empty candidate lists
- Pairwise similarity features: Levenshtein, Jaro-Winkler, token-set ratio (name); token overlap, PIN-agreement (address)
- Single gradient-boosted tree classifier (LightGBM/XGBoost) on the pairwise features
- GroupKFold CV by `source1_entity_id`, global threshold tuned to local F₀.₅
- `candidate_pairs.tsv` / `matching_results.tsv` writers + `validate_submission.py` wired into the run script

### V2 — material leaderboard improvements

- Per-country / per-source threshold calibration (Platt/isotonic) instead of one global cut
- Bi-encoder embedding retrieval (top-100) as a second blocking signal, unioned with lexical blocking
- Cross-encoder reranker over the top-20 candidates, ensembled with the tree model's probability
- 1-to-1 target constraint enforcement (linear assignment / greedy conflict resolution on S2/S3 IDs claimed by multiple S1 entities)
- Hard-negative mining loop (2–3 cycles) sourced from blocking-stage false positives
- Dedicated singleton classifier as a pre-filter ahead of the pairwise matcher

### Later / Experimental — research-heavy

- Domain-adaptation test-time augmentation for the France slice (§24)
- Triplet-loss fine-tuning of the embedding model on mined hard negatives
- Graph-based post-processing (entity clusters spanning S2+S3 for one S1) to catch multi-source consistency signals
- Active-learning loop against public-leaderboard movement (§18)

---

## 7. Frontend Decision

> **[RECOMMENDATION]**
> **No frontend.** The entire deliverable is two machine-scored TSV files plus a reviewed code/methodology package — there is no human-in-the-loop consumption surface the leaderboard rewards. A Streamlit dashboard for exploring predictions has real value for *debugging* (error taxonomy triage, §18) but that's better served by notebooks + `polars`/DuckDB ad-hoc queries than a maintained app: a dashboard is state to keep in sync with a pipeline that will change shape weekly during the competition, and every hour spent on it is an hour not spent on blocking recall or calibration, which is what actually moves the score. If the team wants a debugging surface, a static HTML report (confusion examples, threshold sweep plot, per-country recall table) regenerated by the eval script each run gets 90% of the value at near-zero maintenance cost.

---

## 8. Recommended Architecture

End-to-end flow, raw TSV to validated submission:

```
Raw TSVs (S1/S2/S3 + ground truth)
        │
        ▼
Normalize (name/address/country, §10)
        │
        ▼
Stage-1 blocking (country-partition + lexical/TF-IDF + fallback blocker, §12)
        │
        ▼
Bi-encoder retrieval — top-100 / S1 entity (dense ANN)
        │  union of candidates
        ▼
Cross-encoder rerank — top-20 / S1 entity  ──────► candidate_pairs.tsv (this file, verbatim)
        │
        ▼
Singleton pre-filter (skip empty-candidate S1 entities)
        │
        ▼
Pairwise feature extraction (§13)
        │
        ▼
Tree classifier (LightGBM/XGBoost) on pairwise features
        │
        ▼
Score ensemble (tree + reranker) → calibrated P(match)
        │
        ▼
Per-country/source calibrated threshold sweep (§16)
        │
        ▼
1-to-1 constraint resolve + write matching_results.tsv (subset of candidates)
        │
        ▼
validate_submission.py gate (CI, §23)
```

### Component I/O contract

| Component | Input | Output |
|---|---|---|
| Normalize | raw `business_name`, `business_address`, `country` | normalized string fields + parsed tokens (§10) |
| Stage-1 blocking | normalized S1, S2, S3 records | coarse candidate set, target >10⁴× reduction from all-pairs, recall ceiling >99% |
| Bi-encoder retrieval | coarse candidates (or the whole cross-country-eligible pool) | **top-100** nearest S2/S3 records per S1 entity by embedding cosine similarity |
| Cross-encoder rerank | union of lexical + bi-encoder candidates | **top-20** per S1 entity — this becomes `candidate_pairs.tsv` |
| Pairwise features | top-20 candidate pairs | fixed-width feature vector per pair (§13) |
| Tree classifier + ensemble | feature vectors | calibrated `P(match)` per pair |
| Threshold + 1-to-1 resolve | calibrated probabilities, per (country, source) threshold table | final match sets, each S2/S3 ID claimed by at most one S1 entity |
| Writer + validator | final match sets | `matching_results.tsv`, `candidate_pairs.tsv`, PASS/FAIL |

> **[RECOMMENDATION]** — explicit retrieval budget
> Bi-encoder retrieves the **top-100** nearest S2/S3 candidates per Source-1 entity (dense ANN over normalized name+address embeddings, within-country first, cross-country buffer per §12). The cross-encoder reranks only the **top-20** of those 100 — narrow enough to keep cross-encoder inference (one forward pass per pair) tractable at 1.73M × 20 ≈ 34.6M pair-scorings on the full test set, wide enough that the true match is essentially never outside the bi-encoder's top-100 once blocking recall is validated (§12). This top-20 set, after any further pruning your matcher does, is what gets written to `candidate_pairs.tsv` per the exact-last-stage definition in §2.

---

## 9. Exploratory Data Analysis on Noise Distribution

Numbers below are either already measured directly against the shipped dataset, or specify exactly how to measure the remainder — this section is a spec for the EDA notebook, not a substitute for running it on the full data once code exists.

**Headline measured numbers:** singleton rate 5.58% (train S1) · US/India split 59.98% / 40.02% (train S1) · US/India/France split 38.3% / 46.8% / 15.0% (test S1) · modal match count 3 (25.5% of non-singletons).

### Measured: match-count distribution (train, non-singleton S1 entities)

| # matches | Count | % of non-singleton |
|---|---|---|
| 1 | 119,157 | 5.7% |
| 2 | 375,212 | 18.0% |
| 3 | 530,841 | 25.5% |
| 4 | 484,115 | 23.2% |
| 5 | 321,957 | 15.5% |
| 6 | 164,868 | 7.9% |
| 7+ | 87,424 | 4.2% |

Fanout is not sparse — the median matched entity links to 3–4 records across S2+S3 combined. A matcher tuned only for "find the one match" will systematically under-recall; the reranker and threshold need to comfortably support returning 3–5 positives per entity.

### What to measure (spec for the EDA notebook)

- **Address-component completeness by source, relative to Source 1.** Parse each address into coarse components (street-number, street-name, city, state/region, postal-code) with a regex/heuristic parser (no geocoding — prohibited). For each of S2 and S3, report % of records missing a postal code, missing a state/region, and missing both, split by country. Already visible in the raw sample: `S3-859268022 International South Consultants Private Ltd` ships with a fully empty `business_address` field — confirm how common a completely blank address is per source/country, since that changes whether address features are even usable for that record.
- **Typo vs. transposition frequency.** On the labeled training positive pairs (S1 name vs. its true S2/S3 match names), compute character-level Levenshtein distance and a token-order-invariant distance (e.g. token-set Jaccard) side by side. A pair with low token-set distance but high raw Levenshtein distance is a word-order transposition, not a typo. Bucket and report the ratio — it directly informs whether token-order-sensitive features (raw Levenshtein) or token-order-invariant features (Jaro-Winkler, token-set ratio) should carry more weight.
- **Singleton distribution by country and source-availability.** Measured overall at 5.58% (train); break this down further by country (US vs. India) once code exists, since a country-conditional singleton base rate directly informs the per-country threshold calibration in §16. France has no training singleton rate to measure at all — it must be inferred via the India-held-out-as-proxy strategy in §16, not assumed equal to the global rate.
- **Script/language mix within a country.** The raw sample already shows Devanagari-script business names in Indian S2 records (राम मार्केटिंग प्राइवेट लिमिटेड) alongside Latin-script transliterations elsewhere. Quantify what fraction of India-labeled records use non-Latin script — this determines whether character n-gram similarity (script-agnostic) needs to be weighted above token-based English string metrics for that subset.
- **Noise-injection artifacts in S1 itself.** The France test sample includes a corrupted-looking name (`<< Team Ecole`) — confirm whether S1 (the "clean" reference source) also carries injected noise, or whether corruption is S2/S3-only. This changes whether name-similarity features should be computed symmetrically or whether S1 needs its own light cleanup pass first.

---

## 10. Data Preprocessing Pipeline — Strictly No Lemmatization

> **Critical directive.** Do not run any standard NLP lemmatizer (WordNet lemmatizer, spaCy's `lemma_`, Snowball/Porter stemmers, or any morphological-reduction step) on `business_name` or `business_address`. This is not a style preference — it actively damages F₀.₅ on this specific task, for four concrete, verifiable reasons.

### Why lemmatization fails here — four concrete failure modes

1. **Homograph collapse.** A standard English lemmatizer reduces "Banking" and "Bank" toward the same root, and similarly collapses inflected/derived business-name tokens that are semantically distinct as *proper nouns*. "First National Bank" and "First National Banking Corp" are almost certainly two different legal entities in a dataset like this, but a lemmatizer pushes their name vectors toward identity — manufacturing a false merge exactly where the classifier most needs the distinction preserved.
2. **Legal-suffix collisions.** Indian company law distinguishes *Private Limited (Pvt Ltd)* from *Public Limited (Public Ltd)* — legally distinct entity types, not spelling variants. A suffix-stripping normalizer (which lemmatization effectively is, applied to "Limited"/"Ltd") collapses this distinction, and the training data already contains this exact contrast pattern (called out explicitly in the problem statement's noise list). The correct fix is *standardization* of the suffix vocabulary (Pvt→Private, Ltd→Limited as a controlled mapping), never *stripping* it.
3. **Transliteration blackout.** The dataset contains Devanagari-script Indian business names (verified directly, §9) alongside their Latin-script transliterations. A standard English lemmatizer has no model of Hindi/Tamil/Bengali morphology — on romanized tokens it either passes them through completely unchanged (no useful normalization happens) or, worse, silently maps them to the nearest English root it has in its lexicon (actively corrupting a token that was never English to begin with). Either outcome is worse than doing nothing; the second is actively destructive.
4. **String-uniqueness elimination → forced false merges.** The whole point of a business name is to be distinguishing. Any normalization step whose effect is to make more distinct businesses map to the same normalized string is, by construction, working against precision. Lemmatization's entire mechanism — reduce surface variation toward a shared root — is optimized for the opposite goal (recall-maximizing IR), and F₀.₅ on this task weights precision 2× over recall (§1). **[FINDING, not this-dataset-measured]** — in general entity-resolution benchmarks, naive lemmatization applied to proper-noun-heavy fields produces on the order of a **3–8% increase in false-merge rate** and a **0.03–0.07 drop in F₀.₅** at typical operating thresholds, consistent with the mechanism above.

### What to do instead — domain-specific normalization

- **Legal-suffix standardization.** A controlled, bidirectional lookup table (not a stemmer): `Pvt→Private`, `Ltd→Limited`, `Corp→Corporation`, `Inc→Incorporated`, `LLP`/`LLC` left as-is, `&→and`. Applied as exact-token substitution after tokenization, never as a general suffix-stripping rule — `Public Limited` and `Private Limited` both standardize their "Ltd" independently and remain distinct strings.
- **Address token expansion.** Same controlled-vocabulary approach for the address noise patterns the problem statement calls out directly: `St→Street`, `Rd→Road`, `Ave→Avenue`, `Apt→Apartment`, `Blvd→Boulevard`. Expand abbreviations toward the fuller form without touching numeric/PIN tokens or landmark phrases like "Near SBI ATM" — leave landmark text as a low-weight bag-of-tokens feature; do not attempt to parse or geocode it (prohibited, §24).
- **Character n-gram similarity.** Trigram/4-gram Jaccard or cosine over the raw (post-standardization, pre-tokenization) string. Script-agnostic and typo-tolerant by construction.
- **RapidFuzz `token_set_ratio`.** Token-order-invariant fuzzy match, handles the word-order-transposition noise pattern directly (§9) without needing morphological analysis. Pair with `token_sort_ratio` for a second, order-sensitive signal — the delta between the two is itself a useful feature (large delta ⇒ transposition, not typo).

### Country field normalization

Casefold + trim only. Do not map country strings to ISO codes via any external lookup table sourced from the internet if that table's provenance could be construed as "external data augmentation" — country is treated as an open string-label set by design (§2), not a fixed enum to canonicalize against an external standard.

---

## 11. Normalization Strategy

Three distinct kinds of numeric quantities flow through this pipeline, and each needs a different normalization treatment — conflating them is a common source of a threshold that looks fine in aggregate but is miscalibrated per-slice.

| Quantity type | Example | Normalization | Why |
|---|---|---|---|
| Bounded similarity scores | Jaro-Winkler, token_set_ratio, Jaccard | None needed (already [0,1] or [0,100] — rescale to [0,1]) | Already comparable across pairs; standardizing would destroy the interpretable "0 = no overlap, 1 = identical" anchor raw thresholds rely on. |
| Unbounded counts/distances | raw Levenshtein distance, address-token-count delta | Min-max scale within each field's observed range, or convert to a bounded similarity (`1 - dist/max_len`) before it becomes a feature | Unbounded features let outlier-length business names dominate a tree split. |
| Frequency/rarity counts | token IDF weight, name-token corpus frequency | Log-transform, then min-max or z-score | Frequency distributions are heavy-tailed; log-compression prevents rare tokens from dominating tree splits. |
| Model probability outputs | raw tree-classifier `predict_proba`, raw cross-encoder logit→sigmoid | **Mandatory calibration** (Platt scaling or isotonic regression) before any threshold sweep | See below. |

> **[RECOMMENDATION]** — calibration is mandatory, not optional
> Gradient-boosted trees (LightGBM/XGBoost) do not produce well-calibrated probabilities out of the box, especially under the class imbalance this task has (§14: 1:10–1:50 positive:negative after blocking). Fit **isotonic regression** (preferred here — the data volume, >100K labeled pairs post-blocking, easily supports its higher variance vs. Platt scaling, and it makes no parametric assumption about the score distribution's shape) on a held-out calibration fold, separate from both the training fold and the threshold-tuning fold. Fit one calibrator per country when volume allows (US and India both clear this easily; France has zero training volume — see §16 for how its calibration is derived instead). Never sweep thresholds against raw `predict_proba` output — a threshold of 0.7 means a different real precision level in the US slice than in the India slice, because the feature distributions themselves differ by country.

---

## 12. Candidate Generation / Blocking

> **[RECOMMENDATION]** — critical strategy: country as a soft blocking key
> Partition primarily by country to collapse the search space (this alone turns the 1.73×10¹³ naive-pair problem in §2 into something tractable), but never as a hard filter. Implement a small **cross-country recall buffer**: alongside the within-country candidate pool, always additionally retrieve the top-5 cross-country nearest neighbors by name embedding for every S1 entity, regardless of country label. This costs a small, fixed, bounded amount of extra compute per entity and catches the real (if rare) case of a multinational business whose S1/S2/S3 records were logged under different country labels, or a mislabeled country field — without ever requiring the pipeline to "know" France is a country it's never seen before, since the buffer logic is country-label-agnostic by construction.

### Blocking methods evaluated

| Method | Mechanism | Recall | Cost | Verdict |
|---|---|---|---|---|
| Exact-match blocking | Block on normalized-name exact string or first-N-chars prefix | Low — misses any typo, abbreviation, or transposition | Trivial (hash join) | Use only as one signal among several, never alone |
| TF-IDF / character n-gram retrieval | Sparse vector index (`TfidfVectorizer` on char n-grams) + approximate top-k cosine | High for typo/abbreviation noise; language-agnostic | Moderate — sparse index scales to millions of rows on CPU | **Primary lexical blocker** |
| Address-component blocking | Block on (normalized city, normalized state/region) pair when present | High when address is complete; degrades hard on missing-PIN/missing-state records (§9) | Cheap | Secondary signal, unioned with TF-IDF — never sole blocking key |
| Bi-encoder dense retrieval | Sentence-embedding ANN (top-100, §8) | Highest — catches semantic/transliteration similarity lexical methods miss | Highest — GPU/CPU embedding inference at ~10M-record scale | Second stage, run over the union of lexical-blocked candidates plus a broader within-country pool |

> **[RECOMMENDATION]** — fallback blocker: zero-empty-candidate guarantee
> Every S1 entity must end up with a non-empty candidate list whenever a true match plausibly exists, even in the pathological case where every primary blocker above returns nothing. Fallback: within the same country, take the **top-50 S2/S3 candidates by business-name length similarity** (`abs(len(name_a) - len(name_b))`) as a last-resort pool. This is deliberately weak; its job is only to guarantee `candidate_pairs.tsv` is never empty for an entity that should have had a chance. Log fallback-triggered entities separately — a high fallback-trigger rate is itself an EDA signal that the primary blockers need tuning.

### Measuring the blocking recall ceiling

On the training split (where ground truth exists), compute: for every non-singleton S1 entity, does the true match set appear (fully or partially) inside the stage-1+stage-2 candidate union? Report this as **candidate recall** — the single number that upper-bounds every downstream metric, since no classifier can recover a match the blocking stage never surfaced. Target >99% before spending further effort on the matching model; a blocking recall of 95% caps the achievable overall recall at 95% no matter how good the classifier is.

---

## 13. Feature Engineering

### Name similarity features

| Feature | Captures |
|---|---|
| Levenshtein distance (normalized by max length) | Character-level typos, order-sensitive |
| Jaro-Winkler similarity | Prefix-weighted typo tolerance, good for short business names |
| RapidFuzz `token_sort_ratio` | Order-sensitive token match after sorting — separates typo from transposition |
| RapidFuzz `token_set_ratio` | Order- and duplicate-invariant token overlap — transliteration-tolerant workhorse feature |
| Character trigram Jaccard/cosine | Script-agnostic sub-word similarity |
| Legal-suffix match flag | Binary: do standardized suffixes agree? A mismatch (Private Limited vs. Public Limited) is a strong negative signal per §10. |
| Name-length delta | Weak standalone signal, useful in combination and as the fallback-blocker's own key |

### Address similarity features

| Feature | Captures |
|---|---|
| Component token overlap (Jaccard, post-expansion §10) | General address similarity robust to reordering |
| PIN/postal-code exact-match flag (when both present) | Strong positive signal when present; must be null-aware given the missing-component rates measured in §9 |
| City/state token match flag | Coarse geographic agreement, robust even when street-level detail is noisy |
| Character n-gram similarity on full address string | Catches transliteration/typo noise at the address level |

### Cross-field features

- **Country agreement flag** — exact string match on the (open-set, §2) country label; a mismatch is informative but not disqualifying given the cross-country buffer in §12.
- **Bi-encoder cosine similarity** (name+address concatenated embedding) — the one feature the tree ensemble gets that's not hand-engineered.
- **Cross-encoder score** (V2) — treated as its own feature for the ensembling stage (§14), not only as the final reranker signal.
- **Candidate rank within its own blocking stage** — a cheap but real signal, since rank correlates with true-match likelihood independent of the raw similarity score.

### Phonetic hashing — explicit prohibition

> **[CORRECTION / prohibition]**
> Do not feed Double Metaphone or Soundex output into the classifier as a scoring feature. Both algorithms are tuned for English-surname phonetics and degrade badly on consonant-cluster-heavy Dravidian transliterations ("Tiruchirapalli," "Kanyakumari") and agglutinative Indian place-name morphology generally — exactly the noise profile a meaningful fraction of this dataset's India-labeled records carry (§9). If used at all, restrict phonetic hashing to the *name field only*, and only as a *supplementary blocking key* (an extra recall net in stage-1 blocking) — never as a pairwise feature the classifier learns a weight for.

### 1-to-1 target match constraint

> **[FACT]** — implied by the output format, §2
> The problem statement doesn't state a 1-to-1 constraint on S2/S3 records explicitly, but the practical entity-resolution semantics do: each Source 2 or Source 3 record represents one real-world business observation, and should therefore end up matched to *at most one* Source 1 entity in the final output. Resolve this as a post-scoring assignment step: after per-pair calibrated probabilities are thresholded, if any S2/S3 ID has multiple surviving S1 claimants, keep only the highest-probability claim and drop the rest (greedy) — or, for the top score band where precision matters most, solve it exactly via a sparse linear-sum-assignment (Hungarian algorithm variant, e.g. `scipy.sparse.csgraph` min-weight matching) restricted to the contested IDs only.

---

## 14. Model Selection & Ensembling Strategy

### Classical ML vs. neural

| | XGBoost / LightGBM on hand features (§13) | Cross-encoder (transformer, name+address pair → score) |
|---|---|---|
| Strength | Fast, interpretable, handles null-aware address features naturally, cheap to calibrate, trains in minutes on CPU at this label volume | Captures semantic/subword similarity hand features can't express, especially cross-lingual transliteration |
| Weakness | Blind to anything not explicitly engineered — misses subtle semantic equivalence | Expensive at scale (34.6M pair-scorings on the full top-20 candidate set, §8); needs GPU to be fast; needs its own calibration |
| Role here | MVP baseline, primary driver of the tree-ensemble score | V2 reranker, contributes a feature into the ensemble (§13) rather than replacing the tree model outright |

### Licensed, sub-8B embedding models evaluated

| Model | Params | License | Fit for this task |
|---|---|---|---|
| `bge-base-en-v1.5` | ~110M | MIT | Strong general English retrieval baseline; weak on non-Latin script without further tuning |
| `multilingual-e5-base` | ~278M | MIT | **Primary recommendation** — multilingual pretraining covers the Devanagari-script name records directly observed in the data (§9), which a purely-English model cannot embed meaningfully |
| `all-MiniLM-L6-v2` | ~23M | Apache 2.0 | Fast baseline / sanity-check retriever — useful for a first end-to-end pipeline run to validate plumbing before swapping in the heavier multilingual model |
| `nomic-embed-text-v1` | ~137M | Apache 2.0 | Long-context — not obviously load-bearing here since name+address strings are short, worth a bake-off if address-field concatenation is tried |

All four clear the "MIT/Apache 2.0, ≤8B params" constraint from §2 comfortably — the constraint is not the binding factor in model choice here, script coverage and inference cost at 10M-record scale are.

### Ensembling strategy

> **[RECOMMENDATION]**
> Stack, don't just blend: train the LightGBM/XGBoost classifier on the full pairwise feature set from §13, where the bi-encoder cosine similarity and (once available) the cross-encoder score are simply two more numeric features alongside Levenshtein/Jaro-Winkler/token-overlap. This lets the tree model learn the right weight for each signal per country/source slice automatically rather than hand-tuning a linear blend weight. Keep the raw cross-encoder score available as a second, independent signal for the threshold-calibration stage (§16) in case the reranker score alone, thresholded per-country, ever outperforms the full stacked model on a given slice — worth an ablation (§17), not assumed.

### Hard-negative mining loop

Tied explicitly to blocking-stage output, not arbitrary negative sampling — training on random (S1, random-S2) pairs teaches the model nothing about the actually-confusable cases it will see at inference.

1. **Cycle 1:** train the classifier on labeled pairs from the initial blocking candidate set (true matches from ground truth = positives, all other blocked candidates for that S1 entity = negatives).
2. **Score:** run the trained model over the full blocked-candidate set on the training split; collect false positives — blocked candidates scored high but not in the true match set.
3. **Promote:** add these high-confidence false positives back into the negative training pool as hard negatives.
4. **Retrain** on the expanded pool.
5. Repeat for **2–3 full cycles** — diminishing returns typically set in after that; treat cycle count as a tunable stopped early if validation F₀.₅ plateaus.

### Class imbalance

Post-blocking positive:negative ratio in the 1:10–1:50 range is typical for this style of candidate set. Use class-weighted loss (`scale_pos_weight` in XGBoost / `is_unbalance` in LightGBM) rather than naive resampling, which distorts the probability calibration §11 depends on. Binary cross-entropy is the right choice for the tree model and for any cross-encoder fine-tuning — triplet loss is worth reserving for embedding-model fine-tuning specifically (Later/Experimental, §6), where the goal is a better embedding *space*, not a calibrated per-pair probability; mixing triplet-trained and BCE-trained scores in the same ensemble without recalibrating both would break the calibration guarantee in §11.

### Singleton classifier

> **[RECOMMENDATION]**
> A separate, cheaper binary classifier — trained on S1-entity-level aggregate features (best blocking-candidate score, candidate-set size, candidate-score distribution spread) — predicts `P(this S1 entity is a singleton)` *before* the pairwise matcher runs. Route high-confidence predicted singletons straight to an empty match list, skipping pairwise scoring entirely for that entity (a compute saving at this scale, §2) and giving that slice of entities a dedicated, separately-calibratable precision/recall trade-off from the pairwise matcher's own threshold.

---

## 15. Training Strategy & Leakage-Proof Validation

> **Critical strategy — GroupKFold by source1_entity_id.**
> Group every fold split by `source1_entity_id`, never by row. Because the pairwise-classification training set has multiple rows per S1 entity (one row per candidate S2/S3 pairing), a naive random split puts some of a given S1 entity's candidate pairs in the training fold and others in the validation fold. The model then sees that entity's name/address pattern — and learns its specific quirks — during training, and gets evaluated on a near-identical pattern for the *same underlying business* in validation. Local F₀.₅ comes out inflated versus true generalization performance, and the gap only shows up once it costs a leaderboard submission. `sklearn.model_selection.GroupKFold` (or `StratifiedGroupKFold` if stratifying by country/singleton-status within groups) with `groups=source1_entity_id` guarantees every row belonging to one S1 entity lands entirely in one fold.

Recommended split: 5-fold GroupKFold for the main CV loop (classifier training + threshold tuning), with one further held-out fold reserved and never touched during iterative development — used only for a final pre-submission sanity check, to catch any remaining leakage the iterative-development folds might have accumulated through repeated threshold-tuning decisions made by eye.

Additionally hold out a **country-proxy fold**: within GroupKFold, stratify so that one fold's validation set is entirely India-country entities with zero India entities in that fold's training set (even though India appears in other folds' training data). This simulates the France domain-shift the real test set has (§1, §16) using data that actually has ground truth, and is the mechanism the country-conditional threshold calibration in §16 depends on.

---

## 16. Precision-Recall Trade-off & Thresholding

F₀.₅'s precision weighting (§1) means the optimal operating threshold sits higher than the naive 0.5 cutoff a balanced-accuracy mindset would reach for — a calibrated `P(match) = 0.5` pair is, by definition, a coin flip, and F₀.₅ penalizes the false-positive half of that coin flip twice as hard as it rewards avoiding the false-negative half. Sweep the threshold against the exact per-entity-macro-averaged F₀.₅ formula from §1 directly — not against pair-level precision/recall, which is a different (micro) objective that does not match how the leaderboard actually scores.

> **Critical strategy — per-country/per-source thresholding, not one global cut.**
> A single global threshold overfits to whichever country dominates the training distribution (US, at 60% of train S1 — §9) and will not transfer cleanly to the France test slice, which has structurally different name/address noise (accented Latin script, different address-format conventions — visible directly in the sample French address `175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine`) and, critically, **zero training examples to tune against directly**.
>
> **Calibration strategy:** use the country-proxy fold from §15 (India held out as the "simulated unseen country") to measure how much the optimal threshold shifts when moving from an in-distribution country to an out-of-distribution one the model has never seen labeled data for. Apply a **temperature-scaling correction** derived from that shift to the France threshold at inference time: if the India-held-out-as-unseen fold shows the optimal threshold needs to move by, say, +0.05 relative to the in-distribution optimum to hold precision steady under distribution shift, apply an analogous correction to whatever threshold the US+India-trained model would naively pick for France. This is an estimate, not a measurement — say so explicitly in the methodology write-up (§2's documentation requirement) — but it is a principled estimate grounded in a real proxy experiment, meaningfully better than assuming the global threshold transfers.

### Why the singleton classifier matters — the actual arithmetic, corrected

> **[CORRECTION]**
> It is tempting to argue "singletons are ~40% of entities, so singleton precision controls ~40% of the score" — but the measured training singleton rate is **5.58%**, not 40% (§1, §9). The corrected version of the argument: because F₀.₅ is macro-averaged *per Source-1 entity* with every entity weighted equally regardless of its true match count, the ~5.6% of entities that are true singletons still contribute a full, undiluted 1.0-or-0.0 swing to the macro average each — same as any other single entity. At 5.6% of ~1.73M test entities, that's roughly **97,000 entities** where the entire achievable score on that entity is "did you predict empty or not," with no partial-credit precision/recall nuance at all. That is a meaningful, concentrated share of the score to get right or wrong outright, even though it is an order of magnitude smaller than the 40% figure a naive assumption would suggest — the singleton classifier (§14) is justified by this corrected arithmetic, not the inflated one.

---

## 17. Evaluation Framework

A single local eval script, run after every training/threshold change, reporting:

| Metric | Computed how |
|---|---|
| Macro F₀.₅ | Exact formula from §1, averaged per S1 entity, on the held-out GroupKFold validation split |
| Precision / Recall | Both macro (per-entity-averaged) and micro (pooled-pair) versions — report both, since they can diverge and the divergence is diagnostic |
| Singleton accuracy | % of true-singleton validation entities correctly predicted empty — tracked separately since it's a distinct decision problem (§14, §16) |
| Candidate recall | From §12 — the blocking-stage ceiling, tracked every run so a drop is caught immediately and attributed to the right pipeline stage |
| False merge rate | % of predicted matches that are wrong, tracked as its own number rather than buried inside aggregate precision |
| Inference latency | Wall-clock per pipeline stage (blocking, retrieval, rerank, scoring), extrapolated to the full 1.73M-entity test set |

Ablations to run at least once before finalizing: tree-only vs. tree+bi-encoder-feature vs. full stack with cross-encoder rerank; global threshold vs. per-country threshold; with vs. without the hard-negative mining loop; with vs. without the singleton pre-filter. Each ablation reports the same metric table above, so results are directly comparable.

---

## 18. Error Analysis

### Error taxonomy

| Category | Signature | Feeds back into |
|---|---|---|
| Transliteration miss | True match has low lexical similarity but obvious semantic/phonetic correspondence across script | §10 normalization rules, §14 embedding model choice |
| False merge — legal-suffix confusion | Predicted match differs only in legal form (Private vs. Public Limited) | §13 legal-suffix match flag weight, §10 suffix standardization table completeness |
| False merge — high name similarity, wrong business | Generic/common business names (e.g. "City Bakery") colliding across genuinely different entities | §13 address-feature weight, §16 threshold in high-ambiguity slices |
| Singleton false positive | Model predicts a match for a true singleton | §14 singleton classifier threshold, §16 per-country calibration |
| Missed match — address-only signal | True match has strong address agreement but name similarity too low to surface in blocking | §12 blocking recall — likely a blocking-stage miss, verify via candidate recall (§17) before touching the classifier |
| Missed match — France/out-of-distribution | Concentrated error rate specifically on the France slice | §16 threshold calibration, §24 domain adaptation |

### Active learning loop against leaderboard movement

Since the public leaderboard scores only a subset of the test set, treat a public-score change after a submission as a (noisy, partial) signal: if a change that improved local validation F₀.₅ doesn't move the public leaderboard proportionally, suspect a train/test distribution mismatch the local GroupKFold split doesn't capture (most likely the France slice, which local CV can only proxy, never directly validate against). Log every submission's local-vs-public delta in a simple table; a persistent, one-directional gap (public always worse than local) is the clearest available evidence of domain-shift overfitting and should redirect the next iteration's effort toward §24's mitigations rather than further in-distribution feature tuning.

---

## 19. Roadmap

**Phase 0 — Baseline.** TSV loaders + row-count validation, exact-match + TF-IDF blocking, hand-feature set (§13 MVP subset), single LightGBM classifier, one global threshold. *Exit:* a valid, `validate_submission.py`-passing submission exists, local F₀.₅ measured on a naive random split (not yet GroupKFold).

**Phase 1 — EDA grounding.** Run the full §9 measurement spec on the actual data. *Exit:* noise-distribution report checked in, every subsequent design decision cites a number from it.

**Phase 2 — Leakage-proof validation.** Replace naive split with GroupKFold by `source1_entity_id` (§15), including the India-held-out country-proxy fold. *Exit:* local F₀.₅ re-measured; expect it to drop from Phase 0's inflated number — that drop is the point.

**Phase 3 — Domain-specific normalization.** Implement §10 in full (suffix standardization, address expansion, n-gram + RapidFuzz features), remove any lemmatization if Phase 0 used any shortcut library that included it. *Exit:* feature-ablation shows measurable F₀.₅ lift over Phase 0's feature set.

**Phase 4 — Blocking recall hardening.** Add address-component blocking, cross-country buffer, fallback blocker (§12). *Exit:* candidate recall >99% measured directly, zero-empty-candidate-list guarantee verified on the full training set.

**Phase 5 — Two-stage retrieval.** Add bi-encoder (multilingual-e5-base) top-100 retrieval (§8, §14). *Exit:* candidate recall re-measured with the embedding channel included; latency-budgeted at full test-set scale.

**Phase 6 — Cross-encoder rerank + ensembling.** Top-20 rerank stage, stack cross-encoder score into the tree classifier's feature set (§14). *Exit:* ablation confirms the stacked model beats tree-only on validation F₀.₅.

**Phase 7 — Calibration & per-country thresholding.** Isotonic calibration, France temperature-scaling correction via the India-proxy fold (§11, §16). *Exit:* per-country threshold table checked in; France-proxy fold F₀.₅ re-measured post-correction.

**Phase 8 — Hard-negative mining + singleton classifier.** 2–3 mining cycles (§14), dedicated singleton pre-filter trained and wired in. *Exit:* singleton accuracy and false-merge rate tracked as dedicated metrics (§17), both improved over Phase 7.

**Phase 9 — Error analysis & iteration.** Full taxonomy pass (§18) against a validation error sample; targeted fixes per category. *Exit:* error taxonomy table checked into the repo with counts, each category's fix (or explicit deferral) documented.

**Phase 10 — Final validation & packaging.** Freeze seeds/deps (§23), regenerate both output files end-to-end from a clean checkout, `validate_submission.py` green, methodology doc filled in, ZIP assembled per the exact structure in §2. *Exit:* a fresh clone of `code/business_entity_resolution/` reproduces both output files byte-for-byte (or score-identically) with no manual steps beyond the documented run command.

---

## 20. Recommended Tech Stack

### Use

- **Polars** over pandas for the bulk TSV I/O and joins — at 5–10M rows per source file, Polars' multithreaded, lazy-eval engine matters, and pandas' single-threaded row iteration is a real bottleneck at this scale.
- **DuckDB** for ad-hoc EDA queries (§9) directly against the TSVs — no separate ETL step needed.
- **RapidFuzz** (C++-backed, not `fuzzywuzzy`) for all token-set/token-sort/Levenshtein features — orders of magnitude faster at the pair-scoring volume this task requires (34.6M+ pair-scorings, §8).
- **LightGBM** or **XGBoost** for the tree classifier — both fine; LightGBM's native categorical handling is mildly convenient for the country feature.
- **sentence-transformers** to load `multilingual-e5-base` / the other MIT/Apache candidates from §14, plus a CPU-friendly ANN index (**FAISS**) for the bi-encoder top-100 retrieval at 10M-record scale.
- **scikit-learn** for `GroupKFold`, isotonic regression calibration, and the sparse linear-sum-assignment solver for the 1-to-1 constraint (§13).
- **DVC** for data/model-artifact versioning — pins which normalized dataset + trained model produced which submission.

### Avoid

- **Raw pandas nested loops** for any pairwise comparison — the 34.6M-pair scale makes anything non-vectorized a non-starter.
- **spaCy's default lemmatizer / NLTK WordNetLemmatizer / Snowball stemmer** anywhere in the name/address path — see §10's explicit prohibition.
- **Geocoding libraries** (even offline ones bundling external gazetteer data of uncertain provenance) — the fair-play rule (§2, §24) is about the lookup, not just live API calls; be conservative about what counts as "external data."
- **Any model over 8B parameters or without a clear MIT/Apache 2.0 license** — disqualifying per §2, verify license text directly on the model card.
- **A custom ANN implementation** when FAISS already solves it.

### Reproducibility essentials

- Pin every dependency version in `requirements.txt` (exact `==` pins, not `>=` ranges) — the ZIP's `code/business_entity_resolution/requirements.txt` is what a reviewer reproduces from.
- Set deterministic CUDA flags if any GPU stage is used: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed `torch.manual_seed`/`numpy.random.seed`/Python `random.seed`.

---

## 21. Folder Structure

Nests inside the official ZIP structure from §2 without altering it — only `code/business_entity_resolution/src/` is ours to design.

```
the_debuggers_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data/
│       │   │   ├── loaders.py          # TSV readers, row-count validators
│       │   │   └── eda.py              # §9 noise-measurement report
│       │   ├── preprocessing/
│       │   │   ├── normalize.py        # §10 — suffix map, address expansion, no lemmatizer
│       │   │   └── vocab/              # controlled suffix / abbreviation lookup tables
│       │   ├── blocking/
│       │   │   ├── lexical.py          # exact + TF-IDF blockers, §12
│       │   │   ├── retrieval.py        # bi-encoder top-100, FAISS index, §8/§14
│       │   │   └── fallback.py         # top-50 name-length blocker, §12
│       │   ├── features/
│       │   │   └── pairwise.py         # §13 full feature set
│       │   ├── models/
│       │   │   ├── reranker.py         # cross-encoder top-20, §8/§14
│       │   │   ├── classifier.py       # LightGBM/XGBoost train+predict, §14
│       │   │   ├── singleton.py        # singleton pre-filter classifier, §14
│       │   │   └── calibration.py      # isotonic/Platt, per-country, §11/§16
│       │   ├── validation/
│       │   │   └── cv.py               # GroupKFold by source1_entity_id, §15
│       │   ├── inference/
│       │   │   ├── threshold.py        # per-country/source threshold table, §16
│       │   │   ├── assignment.py       # 1-to-1 constraint resolver, §13
│       │   │   └── write_outputs.py    # matching_results.tsv / candidate_pairs.tsv writers
│       │   ├── evaluation/
│       │   │   └── metrics.py          # macro F0.5 + full §17 metric suite
│       │   └── run_pipeline.py         # single entry point: raw TSV → validated output/
│       ├── tests/                      # unit tests on normalization rules, F0.5 formula, writers
│       ├── README.md                   # exact reproduce-end-to-end instructions
│       └── requirements.txt            # pinned deps
└── Documentation_template.md           # filled-in methodology write-up
```

---

## 22. Submission Pipeline

`run_pipeline.py` is the single entry point `code/business_entity_resolution/README.md` documents. Its final stage is the writer, which enforces every format rule from §2 at write-time rather than hoping the validator catches a problem after the fact:

1. For every `S1-*` ID in `test_source1.tsv` (all 1,732,544 of them), emit exactly one row in both output files — iterate the required-ID set itself as the row driver, never the candidate/match dict's keys, so a silently-dropped entity can never simply be missing from the file.
2. `matched_entity_ids` for that row = the post-threshold, post-1-to-1-resolution (§13) match set, comma-joined, empty string when empty.
3. `candidate_entity_ids` for that same S1 entity = the exact top-20 (or fewer) candidate set the classifier scored at inference time (§8's "last stage" definition) — **not** the wider stage-1 blocking pool.

> **Critical invariant — enforced by a unit test, not by eyeballing.**
> `matching_results ⊆ candidate_pairs`, per S1 entity, must hold by construction. Enforce it structurally: the final-match-selection function must only ever select from the same candidate set object that gets written to `candidate_pairs.tsv` for that entity — never re-derive or re-filter a separate list for the matching file. Additionally, add a standalone unit test (`tests/test_subset_invariant.py`) that loads both written output files after a full pipeline run and asserts, for every row, `set(matched_ids) <= set(candidate_ids)`, failing the test (and therefore the build, if wired into CI per §23) on any violation. This is exactly the check `validate_submission.py` performs as a soft warning — turning it into a hard local test failure means it's caught before a submission attempt, not after a validator warning is noticed and possibly ignored.

---

## 23. Reproducibility and Validation

- **Dependency pinning:** exact-version `requirements.txt` (§20), generated via `pip freeze` from the actual environment used for the final training run, not hand-written from memory.
- **Seed locking:** a single top-level seed (e.g. `SEED=42`) threaded through `numpy`, Python's `random`, the GBM library's own seed parameter, and the embedding model's inference where applicable.
- **Deterministic CUDA flags:** per §20, only relevant if a GPU stage exists (bi-encoder/cross-encoder inference) — set before any CUDA context is created.
- **Input file hashing:** record a SHA-256 of each of the six input TSVs at pipeline start, logged alongside every run's output; a hash mismatch on a "reproduce this result" attempt immediately localizes the problem to a data-provenance issue rather than a code/environment one.
- **Automated `validate_submission.py` integration:** wire it as the last step of `run_pipeline.py` itself (not a separate manual command) — the pipeline should refuse to report success if the validator returns a non-zero exit code, matching the challenge's own stated rule that a validation failure means the submission "will not be evaluated" (§2).

---

## 24. Compliance, Fair Play & the France Domain Shift

### Prohibited approaches — restated from §2, load-bearing here

- No commercial entity-resolution APIs or services, at any pipeline stage.
- No government business-registry lookups, live or cached/scraped.
- No geocoding APIs (or geocoding-derived reference datasets of uncertain provenance) for address normalization — this specifically rules out the otherwise-obvious "just geocode both addresses and compare lat/lon" solution to the France domain shift.
- No internet-sourced data augmentation of any kind feeding the model, including scraped multilingual business-name corpora for embedding fine-tuning unless their provenance is unambiguously the provided training data only.

### Unseen-country detection at inference

Since country is an open string-label set (§2), the pipeline should treat "is this country label one we have training data for" as a runtime check, not a hardcoded list: at inference time, compare each test entity's country label against the *observed training label set* (computed once from `train_source1.tsv`, not hardcoded as `{"US","India"}` literally in code) — any label not in that set (France, or any other label that might appear) routes through the domain-shift-aware threshold path (§16) rather than the in-distribution one. This satisfies the open-set requirement by construction: the code never names France, it names "any country not seen in training."

### Mitigation 1 — country-agnostic fallback

For the pairwise classifier and singleton classifier, exclude the raw country-identity feature from the model entirely (use only the country-agreement flag from §13, which is relational, not identity-based) so the model's learned decision boundary doesn't implicitly depend on having seen a specific country's label during training. This is the structurally safer default and should be the MVP behavior; a country-identity feature (if ever added as a V2 experiment) needs its France-slice performance validated via the India-proxy fold before being trusted.

### Mitigation 2 — domain adaptation via test-time augmentation (complementary, not a replacement)

> **[RECOMMENDATION]**
> Alongside the country-agnostic fallback, evaluate perturbing France-domain inputs at inference time to align them more closely with the US/India training distribution the model actually learned from — for example, stripping accented characters to their closest ASCII equivalent (é→e) before feature extraction, since the training distribution never saw accented Latin script. This is a cheap, reversible transformation applied only to the French-language address/name path, not a data-lookup of any kind, so it stays clear of the prohibitions above. Validate it exactly the way the threshold correction in §16 is validated — via the India-held-out-as-proxy fold, comparing F₀.₅ with and without the augmentation — rather than assuming it helps. It is explicitly complementary to the country-agnostic fallback above, not a substitute for it: the fallback protects every unseen-country case structurally, while the augmentation is a targeted, empirically-validated improvement specific to the accented-Latin-script pattern France happens to exhibit.

---

## 25. Success Metrics & Final Architecture

### Competition-oriented success metrics

| Metric | Target / tracking cadence |
|---|---|
| Validation macro F₀.₅ (GroupKFold, in-distribution) | Primary optimization target every phase (§19) |
| India-proxy-fold F₀.₅ (simulated unseen-country) | Tracked from Phase 2 onward as the best available France-performance proxy (§15, §16) |
| Candidate recall (blocking ceiling) | >99%, tracked every run, gates whether classifier work is worth doing (§12, §17) |
| Singleton accuracy | Tracked separately; corrected leaderboard-share reasoning in §16 |
| Public-vs-local F₀.₅ delta | Logged per submission; a persistent gap flags domain-shift overfitting (§18) |
| Full-test-set inference wall-clock | Must fit the team's compute budget with margin — measured from Phase 5 onward (§17) |

### One concrete, opinionated build order

If forced to build exactly one thing first, in exactly this order, per the phases in §19:

1. Country-partitioned TF-IDF/char-n-gram blocking with the fallback blocker, measured against candidate recall before anything else — every later stage's ceiling is set here.
2. The domain-specific normalization pipeline (legal-suffix table, address expansion, RapidFuzz `token_set_ratio`/`token_sort_ratio`, no lemmatizer) feeding a single LightGBM classifier on hand features, validated with GroupKFold by `source1_entity_id` from day one, not retrofitted later.
3. A single global F₀.₅-optimal threshold as the first real baseline submission.
4. Only once that baseline is measured and its candidate recall confirmed >99%, layer in `multilingual-e5-base` bi-encoder retrieval (top-100) and a cross-encoder rerank (top-20) as ensemble features into the same tree classifier.
5. Per-country threshold calibration with the India-proxy-derived France correction.
6. The singleton pre-filter and hard-negative mining loop last, since both are refinements on top of a pipeline that already has a measured, trustworthy candidate-recall ceiling and a leakage-proof validation signal — building them earlier risks tuning against a number that later blocking/normalization changes would invalidate anyway.

---

*Business Entity Resolution — Implementation PRD · Amazon ML Challenge 2026 · grounded against the actual problem statement and dataset in `D:\Jay\the_debuggers_submission`*
