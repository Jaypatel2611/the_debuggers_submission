# Phase 0 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish PRD §19 Phase 0 — exact+TF-IDF blocking, MVP hand-feature set, one LightGBM classifier, one global threshold, a `validate_submission.py`-passing submission, and a naive-random-split local F₀.₅ number.

**Architecture:** `run_pipeline.py` wires: loaders (done) → normalize → block (exact + TF-IDF, per country) → pairwise features → LightGBM train/predict on a naive row-level split → global threshold sweep on macro F₀.₅ → greedy 1-to-1 assignment → writer (`matching_results.tsv` + `candidate_pairs.tsv`, subset-invariant by construction) → `validate_submission.py` subprocess gate. Same code path serves training-pair construction (train S1 vs train S2/S3, labeled via `train_ground_truth`) and test-time inference (test S1 vs test S2/S3, unlabeled).

**Tech Stack:** Polars (I/O + joins), RapidFuzz (name similarity), scikit-learn (`TfidfVectorizer`, sparse cosine top-k, `train_test_split`), LightGBM (classifier), stdlib (assignment, writer).

**Spec:** `PRD.md` §§2, 9–23 (this plan implements the §19 Phase 0 line item only — no bi-encoder/§8, no cross-encoder/§14 V2, no GroupKFold/§15 (that's Phase 2), no per-country threshold/§16 (Phase 7), no hard-negative mining/§14, no singleton classifier/§14, no fallback blocker/§12 (Phase 4), no full §10 normalization (Phase 3)).

## Global Constraints

- No lemmatization anywhere (PRD §10) — normalization here is lowercase + whitespace/punctuation collapse only, nothing morphological.
- Country is an open-set string label — never hard-code `{"US","India"}`; iterate whatever country values are actually present in the data (France appears only in test).
- No pandas — Polars only (PRD §20).
- `entity_id` prefix (`S1-`/`S2-`/`S3-`) is the only source indicator.
- Seed everything with `SEED = 42` (numpy, Python `random`, LightGBM `random_state`) — PRD §23.
- `matched_entity_ids` ⊆ `candidate_entity_ids` per S1 entity must hold by construction (PRD §22), not by post-hoc filtering.
- Every test `S1-*` ID gets exactly one row in both output files, driven by iterating `test_source1`'s ID column — never by iterating a candidates/matches dict's keys.
- Pin every new dependency in `requirements.txt` with `==` (scikit-learn, lightgbm, scipy already installed — versions below).
- Output files land at `<repo_root>/output/matching_results.tsv` and `<repo_root>/output/candidate_pairs.tsv` (PRD §21 ZIP layout — `output/` is a sibling of `code/`, not inside it).

## Review Focus

- **Singleton entities (5.58% of train S1):** predicting empty for a true singleton scores 1.0, predicting anything scores 0.0. Threshold sweep and F₀.₅ calc must handle the empty-prediction / empty-truth case without divide-by-zero, and must not be swamped by non-singleton rows in a plain accuracy-style metric.
- **A blocked candidate pool can legitimately be empty** for an S1 entity when its country has no same-country S2/S3 rows at all (shouldn't happen in-sample, but must not crash — empty TF-IDF result for a query row is a normal sparse-matmul outcome, not an error).
- **The same S2/S3 ID can be the top candidate for two different S1 entities** — the greedy 1-to-1 assignment (§13) must resolve this by keeping only the highest-probability claim, and must be tested with a constructed collision, not just spot-checked.
- **`matched_entity_ids`/`candidate_entity_ids` empty-list formatting** — must serialize as nothing after the tab (not `"[]"`, not `"None"`), verified by a test that parses the written TSV back with the exact same TSV reader `loaders.py` uses.
- **A test S1 entity whose country never appears in test S2/S3 candidates at all** (pathological but must not silently drop the row) — writer must still emit a row with an empty match list for it, proven by a test with a manufactured orphan-country S1 row.

---

## File Structure

```
code/business_entity_resolution/
├── requirements.txt                          # + scikit-learn, lightgbm, scipy pins
├── src/
│   ├── preprocessing/normalize.py            # NEW — MVP normalize_name/normalize_address
│   ├── blocking/lexical.py                   # NEW — exact + TF-IDF blocker, per-country
│   ├── features/pairwise.py                  # NEW — §13 MVP feature set, one row per candidate pair
│   ├── models/classifier.py                  # NEW — LightGBM train/predict wrapper
│   ├── evaluation/metrics.py                 # NEW — macro/micro F0.5, precision/recall, singleton acc, candidate recall
│   ├── inference/threshold.py                # NEW — global-threshold sweep against macro F0.5
│   ├── inference/assignment.py                # NEW — greedy 1-to-1 resolver
│   ├── inference/write_outputs.py             # NEW — matching_results.tsv / candidate_pairs.tsv writer
│   └── run_pipeline.py                        # NEW — entry point, wires everything, calls validate_submission.py
└── tests/
    ├── test_normalize.py                      # NEW
    ├── test_lexical.py                        # NEW
    ├── test_pairwise.py                       # NEW
    ├── test_metrics.py                        # NEW
    ├── test_threshold.py                      # NEW
    ├── test_assignment.py                     # NEW
    └── test_write_outputs.py                  # NEW
```

---

### Task 1: Normalization (MVP)

**Files:**
- Create: `code/business_entity_resolution/src/preprocessing/normalize.py`
- Test: `code/business_entity_resolution/tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_text(s: str) -> str` — lowercase, collapse whitespace, strip non-alphanumeric except spaces. Used by every later task that touches `business_name`/`business_address`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.normalize import normalize_text

def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_text("  Acme   CORP.  ") == "acme corp"

def test_normalize_strips_punctuation_but_keeps_alnum_and_spaces():
    assert normalize_text("O'Brien & Sons, Ltd.") == "o brien sons ltd"

def test_normalize_empty_string_stays_empty():
    assert normalize_text("") == ""

def test_normalize_does_not_lemmatize():
    # "stores" must stay "stores" -- a lemmatizer would fold it to "store".
    assert normalize_text("Corner Stores") == "corner stores"
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_normalize.py -v` → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/preprocessing/normalize.py
"""MVP text normalization (PRD §19 Phase 0 subset of §10).

Lowercase + whitespace/punctuation collapse only -- NO lemmatization, NO
stemming, NO suffix standardization table (that's Phase 3, PRD §10). This
exists only so blocking/feature code has a stable string to key/compare on.
"""
from __future__ import annotations

import re

_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9 ]+")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = _NON_ALNUM_SPACE.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_normalize.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/preprocessing/normalize.py code/business_entity_resolution/tests/test_normalize.py
git commit -m "feat: add MVP text normalization for Phase 0 blocking/features"
```

---

### Task 2: Lexical blocking — exact + TF-IDF, per country

**Files:**
- Create: `code/business_entity_resolution/src/blocking/lexical.py`
- Test: `code/business_entity_resolution/tests/test_lexical.py`

**Interfaces:**
- Consumes: `normalize_text` from Task 1.
- Produces: `generate_candidates(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame, top_k: int = 20) -> pl.DataFrame` returning columns `source1_entity_id, candidate_id, country, blocking_rank, tfidf_score, exact_match` (one row per S1×candidate pair). `s1`/`s2`/`s3` each have columns `entity_id, business_name, business_address, country`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lexical.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from src.blocking.lexical import generate_candidates

def _df(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")

def test_exact_match_always_included():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "US"), ("S2-2", "Totally Different Biz", "9 Elm St", "US")])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=1)
    ids = result.filter(pl.col("source1_entity_id") == "S1-1")["candidate_id"].to_list()
    assert "S2-1" in ids

def test_tfidf_ranks_similar_name_above_dissimilar():
    s1 = _df([("S1-1", "Acme Corporation", "1 Main St", "US")])
    s2 = _df([
        ("S2-1", "Acme Corp", "1 Main St", "US"),
        ("S2-2", "Zzyzx Unrelated Widgets", "9 Elm St", "US"),
    ])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=2).sort("blocking_rank")
    top = result.filter(pl.col("source1_entity_id") == "S1-1").row(0, named=True)
    assert top["candidate_id"] == "S2-1"

def test_never_crosses_country():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "India")])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=5)
    assert result.filter(pl.col("source1_entity_id") == "S1-1").height == 0

def test_no_same_country_candidates_returns_empty_not_crash():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "France")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "US")])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=5)
    assert result.height == 0
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_lexical.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/blocking/lexical.py
"""Exact-match + TF-IDF lexical blocking, partitioned by country (PRD §12
MVP subset for Phase 0 -- no address-component blocking, no bi-encoder, no
cross-country buffer, no fallback blocker: those are Phase 4/5).
"""
from __future__ import annotations

import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocessing.normalize import normalize_text

_CANDIDATE_SCHEMA = ["source1_entity_id", "candidate_id", "country", "blocking_rank", "tfidf_score", "exact_match"]


def _tfidf_topk(s1_names: list[str], cand_names: list[str], top_k: int) -> list[list[tuple[int, float]]]:
    """Per S1 row, up to top_k (candidate_row_index, cosine_score) pairs."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    cand_matrix = vec.fit_transform(cand_names)  # (n_cand, vocab), L2-normalized rows
    s1_matrix = vec.transform(s1_names)  # (n_s1, vocab)
    sims = s1_matrix @ cand_matrix.T  # sparse (n_s1, n_cand), cosine since both L2-normalized
    sims = sparse.csr_matrix(sims)
    out: list[list[tuple[int, float]]] = []
    for row_idx in range(sims.shape[0]):
        start, end = sims.indptr[row_idx], sims.indptr[row_idx + 1]
        cols = sims.indices[start:end]
        vals = sims.data[start:end]
        if len(vals) == 0:
            out.append([])
            continue
        order = vals.argsort()[::-1][:top_k]
        out.append([(int(cols[i]), float(vals[i])) for i in order])
    return out


def generate_candidates(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame, top_k: int = 20) -> pl.DataFrame:
    s2 = s2.with_columns(source=pl.lit("S2"))
    s3 = s3.with_columns(source=pl.lit("S3"))
    candidates = pl.concat([s2, s3], how="vertical")

    rows: list[dict] = []
    for country in s1["country"].unique().to_list():
        s1_country = s1.filter(pl.col("country") == country)
        cand_country = candidates.filter(pl.col("country") == country)
        if s1_country.height == 0 or cand_country.height == 0:
            continue

        s1_ids = s1_country["entity_id"].to_list()
        s1_names_norm = [normalize_text(n) for n in s1_country["business_name"].to_list()]
        cand_ids = cand_country["entity_id"].to_list()
        cand_names_norm = [normalize_text(n) for n in cand_country["business_name"].to_list()]

        topk = _tfidf_topk(s1_names_norm, cand_names_norm, top_k)
        for i, s1_id in enumerate(s1_ids):
            seen: set[str] = set()
            rank = 0
            # exact match first (forced in, regardless of TF-IDF rank)
            for j, cand_name_norm in enumerate(cand_names_norm):
                if cand_name_norm == s1_names_norm[i] and cand_ids[j] not in seen:
                    rows.append({
                        "source1_entity_id": s1_id, "candidate_id": cand_ids[j], "country": country,
                        "blocking_rank": rank, "tfidf_score": 1.0, "exact_match": True,
                    })
                    seen.add(cand_ids[j])
                    rank += 1
            for j, score in topk[i]:
                cand_id = cand_ids[j]
                if cand_id in seen or rank >= top_k:
                    continue
                rows.append({
                    "source1_entity_id": s1_id, "candidate_id": cand_id, "country": country,
                    "blocking_rank": rank, "tfidf_score": score, "exact_match": False,
                })
                seen.add(cand_id)
                rank += 1

    if not rows:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_id": pl.Utf8, "country": pl.Utf8,
            "blocking_rank": pl.Int64, "tfidf_score": pl.Float64, "exact_match": pl.Boolean,
        })
    return pl.DataFrame(rows).select(_CANDIDATE_SCHEMA)
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_lexical.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/blocking/lexical.py code/business_entity_resolution/tests/test_lexical.py
git commit -m "feat: add exact+TF-IDF per-country blocking (PRD §12 Phase 0 subset)"
```

---

### Task 3: Pairwise features (§13 MVP subset)

**Files:**
- Create: `code/business_entity_resolution/src/features/pairwise.py`
- Test: `code/business_entity_resolution/tests/test_pairwise.py`

**Interfaces:**
- Consumes: candidate rows from Task 2 (`source1_entity_id, candidate_id, country, blocking_rank, tfidf_score, exact_match`), plus the raw `s1`/`s2`/`s3` frames for name/address lookup.
- Produces: `build_features(candidates: pl.DataFrame, s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame` — same rows as `candidates`, plus feature columns: `lev_ratio, jaro_winkler, token_sort_ratio, token_set_ratio, char_trigram_jaccard, name_len_delta, country_match, addr_token_jaccard, blocking_rank, tfidf_score`. `FEATURE_COLUMNS` module-level constant listing the exact column names the classifier trains on (Task 4 imports this).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pairwise.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from src.features.pairwise import build_features, FEATURE_COLUMNS

def _df(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")

def test_identical_pair_scores_max_similarity():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "US")])
    s3 = _df([])
    cands = pl.DataFrame({
        "source1_entity_id": ["S1-1"], "candidate_id": ["S2-1"], "country": ["US"],
        "blocking_rank": [0], "tfidf_score": [1.0], "exact_match": [True],
    })
    feats = build_features(cands, s1, s2, s3)
    row = feats.row(0, named=True)
    assert row["lev_ratio"] == pytest.approx(1.0)
    assert row["token_set_ratio"] == pytest.approx(100.0)
    assert row["country_match"] == 1
    assert row["name_len_delta"] == 0

def test_country_mismatch_flagged():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s3 = _df([("S3-1", "Acme Corp", "1 Main St", "India")])
    s2 = _df([])
    cands = pl.DataFrame({
        "source1_entity_id": ["S1-1"], "candidate_id": ["S3-1"], "country": ["US"],
        "blocking_rank": [0], "tfidf_score": [0.5], "exact_match": [False],
    })
    feats = build_features(cands, s1, s2, s3)
    assert feats.row(0, named=True)["country_match"] == 0

def test_feature_columns_are_all_numeric():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([("S2-1", "Beta Inc", "2 Oak Ave", "US")])
    s3 = _df([])
    cands = pl.DataFrame({
        "source1_entity_id": ["S1-1"], "candidate_id": ["S2-1"], "country": ["US"],
        "blocking_rank": [0], "tfidf_score": [0.1], "exact_match": [False],
    })
    feats = build_features(cands, s1, s2, s3)
    for col in FEATURE_COLUMNS:
        assert feats[col].dtype in (pl.Float64, pl.Int64, pl.Int32, pl.Boolean)
```

(needs `import pytest` at top alongside `polars`).

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_pairwise.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/features/pairwise.py
"""Pairwise similarity features -- PRD §13 MVP subset for Phase 0.

Excluded here on purpose (later phases, not missing by accident):
legal-suffix match flag (needs §10's suffix table, Phase 3), PIN/postal
exact-match + city/state flags (need address-component parsing, Phase 3),
bi-encoder cosine + cross-encoder score (Phase 5/6), candidate-rank is kept
(cheap, already produced by blocking).
"""
from __future__ import annotations

import polars as pl
from rapidfuzz import fuzz

from ..preprocessing.normalize import normalize_text

FEATURE_COLUMNS = [
    "lev_ratio", "jaro_winkler", "token_sort_ratio", "token_set_ratio",
    "char_trigram_jaccard", "name_len_delta", "country_match",
    "addr_token_jaccard", "blocking_rank", "tfidf_score",
]


def _trigrams(s: str) -> set[str]:
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s} if s else set()


def _trigram_jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_features(candidates: pl.DataFrame, s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    if candidates.height == 0:
        return candidates.with_columns(**{c: pl.lit(None) for c in FEATURE_COLUMNS})

    s1_lookup = dict(zip(s1["entity_id"].to_list(), zip(s1["business_name"].to_list(), s1["business_address"].to_list(), s1["country"].to_list())))
    cand_source = pl.concat([s2, s3], how="vertical")
    cand_lookup = dict(zip(cand_source["entity_id"].to_list(), zip(cand_source["business_name"].to_list(), cand_source["business_address"].to_list(), cand_source["country"].to_list())))

    rows = []
    for rec in candidates.iter_rows(named=True):
        name_a, addr_a, country_a = s1_lookup[rec["source1_entity_id"]]
        name_b, addr_b, country_b = cand_lookup[rec["candidate_id"]]
        na, nb = normalize_text(name_a), normalize_text(name_b)
        aa, ab = normalize_text(addr_a), normalize_text(addr_b)
        rows.append({
            **rec,
            "lev_ratio": fuzz.ratio(na, nb) / 100.0,
            "jaro_winkler": fuzz.WRatio(na, nb) / 100.0,
            "token_sort_ratio": fuzz.token_sort_ratio(na, nb),
            "token_set_ratio": fuzz.token_set_ratio(na, nb),
            "char_trigram_jaccard": _trigram_jaccard(na, nb),
            "name_len_delta": abs(len(na) - len(nb)),
            "country_match": int(country_a == country_b),
            "addr_token_jaccard": _token_jaccard(aa, ab),
        })
    return pl.DataFrame(rows)
```

*(Note: RapidFuzz has no standalone Jaro-Winkler in the pinned version's public API surface used elsewhere in this repo — `fuzz.WRatio` is used as the prefix-weighted stand-in; if `rapidfuzz.distance.JaroWinkler.normalized_similarity` is available in the pinned `rapidfuzz==3.14.6`, use that instead for a literal Jaro-Winkler and update the test's expected column semantics accordingly — verify with `python -c "from rapidfuzz.distance import JaroWinkler; print(JaroWinkler.normalized_similarity('a','a'))"` before finalizing.)*

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_pairwise.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/features/pairwise.py code/business_entity_resolution/tests/test_pairwise.py
git commit -m "feat: add PRD §13 MVP pairwise feature set"
```

---

### Task 4: Evaluation metrics (§17)

**Files:**
- Create: `code/business_entity_resolution/src/evaluation/metrics.py`
- Test: `code/business_entity_resolution/tests/test_metrics.py`

**Interfaces:**
- Produces: `f_beta(precision: float, recall: float, beta: float = 0.5) -> float`; `macro_f05(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> float`; `evaluate(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> dict` returning `{macro_f05, macro_precision, macro_recall, micro_precision, micro_recall, singleton_accuracy}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.evaluation.metrics import f_beta, macro_f05, evaluate

def test_f_beta_perfect_match_is_one():
    assert f_beta(1.0, 1.0) == pytest.approx(1.0)

def test_f_beta_zero_precision_and_recall_is_zero():
    assert f_beta(0.0, 0.0) == 0.0

def test_macro_f05_true_singleton_predicted_empty_scores_one():
    result = macro_f05({"S1-1": set()}, {"S1-1": set()})
    assert result == pytest.approx(1.0)

def test_macro_f05_true_singleton_predicted_nonempty_scores_zero():
    result = macro_f05({"S1-1": {"S2-1"}}, {"S1-1": set()})
    assert result == pytest.approx(0.0)

def test_macro_f05_weights_precision_over_recall():
    # one false positive alongside the true match: precision 0.5, recall 1.0
    result = macro_f05({"S1-1": {"S2-1", "S2-2"}}, {"S1-1": {"S2-1"}})
    # F0.5 = 1.25*0.5*1.0 / (0.25*0.5 + 1.0) = 0.625/1.125
    assert result == pytest.approx(0.625 / 1.125)

def test_evaluate_reports_singleton_accuracy():
    preds = {"S1-1": set(), "S1-2": {"S2-1"}}
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    result = evaluate(preds, truth)
    assert result["singleton_accuracy"] == pytest.approx(1.0)
    assert result["macro_f05"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_metrics.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/metrics.py
"""PRD §17 evaluation framework: macro F0.5 (the leaderboard metric, §1) plus
the diagnostic metrics table. Predictions/truth are {source1_entity_id: set(matched_ids)}."""
from __future__ import annotations


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def _precision_recall(pred: set[str], truth: set[str]) -> tuple[float, float]:
    if not pred and not truth:
        return 1.0, 1.0
    if not pred:
        return 1.0, 0.0
    if not truth:
        return 0.0, 1.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    return precision, recall


def macro_f05(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    scores = []
    for s1_id, truth_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r = _precision_recall(pred_set, truth_set)
        scores.append(f_beta(p, r, beta=0.5))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> dict:
    precisions, recalls = [], []
    tp_total = pred_total = truth_total = 0
    singleton_total = singleton_correct = 0
    for s1_id, truth_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r = _precision_recall(pred_set, truth_set)
        precisions.append(p)
        recalls.append(r)
        tp_total += len(pred_set & truth_set)
        pred_total += len(pred_set)
        truth_total += len(truth_set)
        if not truth_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
    return {
        "macro_f05": macro_f05(predictions, truth),
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "micro_precision": tp_total / pred_total if pred_total else 1.0,
        "micro_recall": tp_total / truth_total if truth_total else 1.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
    }
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_metrics.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/evaluation/metrics.py code/business_entity_resolution/tests/test_metrics.py
git commit -m "feat: add PRD §17 macro F0.5 and diagnostic metrics"
```

---

### Task 5: LightGBM classifier wrapper

**Files:**
- Create: `code/business_entity_resolution/src/models/classifier.py`
- Modify: `code/business_entity_resolution/requirements.txt` — add `scikit-learn==1.8.0`, `lightgbm==4.7.0`, `scipy==1.17.1`
- Test: covered by Task 3's fixtures + a small direct test below

**Interfaces:**
- Consumes: `FEATURE_COLUMNS` from Task 3.
- Produces: `train_classifier(train_df: pl.DataFrame, label_col: str = "label", seed: int = 42) -> lightgbm.Booster`; `predict_proba(model, df: pl.DataFrame) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from src.models.classifier import train_classifier, predict_proba
from src.features.pairwise import FEATURE_COLUMNS

def _labeled_df(n_pos=20, n_neg=20):
    import random
    random.seed(0)
    rows = []
    for _ in range(n_pos):
        rows.append({**{c: random.uniform(0.8, 1.0) for c in FEATURE_COLUMNS}, "label": 1})
    for _ in range(n_neg):
        rows.append({**{c: random.uniform(0.0, 0.2) for c in FEATURE_COLUMNS}, "label": 0})
    return pl.DataFrame(rows)

def test_train_and_predict_separates_classes():
    df = _labeled_df()
    model = train_classifier(df, seed=42)
    probs = predict_proba(model, df)
    pos_mean = probs[df["label"].to_numpy() == 1].mean()
    neg_mean = probs[df["label"].to_numpy() == 0].mean()
    assert pos_mean > neg_mean
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_classifier.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/models/classifier.py
"""LightGBM pairwise-match classifier -- PRD §14 MVP: one model, no
ensembling, no hard-negative mining loop (Phase 8), no singleton
pre-filter (Phase 8)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from ..features.pairwise import FEATURE_COLUMNS

SEED = 42


def train_classifier(train_df: pl.DataFrame, label_col: str = "label", seed: int = SEED) -> lgb.Booster:
    X = train_df.select(FEATURE_COLUMNS).to_numpy()
    y = train_df[label_col].to_numpy()
    dataset = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
    params = {
        "objective": "binary",
        "is_unbalance": True,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
    }
    return lgb.train(params, dataset, num_boost_round=100)


def predict_proba(model: lgb.Booster, df: pl.DataFrame) -> np.ndarray:
    X = df.select(FEATURE_COLUMNS).to_numpy()
    return model.predict(X)
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_classifier.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/models/classifier.py code/business_entity_resolution/tests/test_classifier.py code/business_entity_resolution/requirements.txt
git commit -m "feat: add LightGBM pairwise classifier (PRD §14 Phase 0 MVP)"
```

---

### Task 6: Global threshold sweep (§16, Phase 0's "one global threshold")

**Files:**
- Create: `code/business_entity_resolution/src/inference/threshold.py`
- Test: `code/business_entity_resolution/tests/test_threshold.py`

**Interfaces:**
- Consumes: `macro_f05` from Task 4.
- Produces: `sweep_threshold(candidates: pl.DataFrame, probs: np.ndarray, truth: dict[str, set[str]], grid: list[float] | None = None) -> tuple[float, float]` — `(best_threshold, best_macro_f05)`. `candidates` has `source1_entity_id, candidate_id` aligned row-for-row with `probs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_threshold.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
import pytest
from src.inference.threshold import sweep_threshold

def test_sweep_finds_threshold_that_separates_classes():
    candidates = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
        "candidate_id": ["S2-1", "S2-2", "S2-3"],
    })
    probs = np.array([0.9, 0.1, 0.05])
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    best_t, best_score = sweep_threshold(candidates, probs, truth, grid=[0.2, 0.5, 0.8])
    assert best_score == pytest.approx(1.0)
    assert 0.2 <= best_t <= 0.8
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_threshold.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference/threshold.py
"""Single global threshold, swept against exact macro F0.5 (PRD §16) --
per-country thresholding is Phase 7, not here."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..evaluation.metrics import macro_f05


def _predictions_at(candidates: pl.DataFrame, probs: np.ndarray, t: float) -> dict[str, set[str]]:
    keep = probs >= t
    preds: dict[str, set[str]] = {s1_id: set() for s1_id in candidates["source1_entity_id"].unique().to_list()}
    for s1_id, cand_id, k in zip(candidates["source1_entity_id"].to_list(), candidates["candidate_id"].to_list(), keep):
        if k:
            preds[s1_id].add(cand_id)
    return preds


def sweep_threshold(
    candidates: pl.DataFrame, probs: np.ndarray, truth: dict[str, set[str]], grid: list[float] | None = None
) -> tuple[float, float]:
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    best_t, best_score = grid[0], -1.0
    for t in grid:
        preds = _predictions_at(candidates, probs, t)
        score = macro_f05(preds, truth)
        if score > best_score:
            best_t, best_score = t, score
    return best_t, best_score
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_threshold.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/inference/threshold.py code/business_entity_resolution/tests/test_threshold.py
git commit -m "feat: add global-threshold sweep against macro F0.5 (PRD §16 Phase 0 subset)"
```

---

### Task 7: Greedy 1-to-1 assignment (§13)

**Files:**
- Create: `code/business_entity_resolution/src/inference/assignment.py`
- Test: `code/business_entity_resolution/tests/test_assignment.py`

**Interfaces:**
- Produces: `resolve_1to1(candidates: pl.DataFrame, probs: np.ndarray, threshold: float) -> dict[str, set[str]]` — for any `candidate_id` claimed by multiple `source1_entity_id`s above threshold, keeps only the highest-probability claim.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_assignment.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from src.inference.assignment import resolve_1to1

def test_contested_id_goes_to_highest_probability_claimant():
    candidates = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "candidate_id": ["S2-1", "S2-1"],
    })
    probs = np.array([0.6, 0.9])
    result = resolve_1to1(candidates, probs, threshold=0.5)
    assert result["S1-2"] == {"S2-1"}
    assert result["S1-1"] == set()

def test_below_threshold_never_claimed():
    candidates = pl.DataFrame({"source1_entity_id": ["S1-1"], "candidate_id": ["S2-1"]})
    probs = np.array([0.1])
    result = resolve_1to1(candidates, probs, threshold=0.5)
    assert result["S1-1"] == set()

def test_uncontested_ids_all_kept():
    candidates = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "candidate_id": ["S2-1", "S2-2"],
    })
    probs = np.array([0.9, 0.8])
    result = resolve_1to1(candidates, probs, threshold=0.5)
    assert result["S1-1"] == {"S2-1", "S2-2"}
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_assignment.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference/assignment.py
"""Greedy 1-to-1 resolver (PRD §13) -- keep only the highest-probability
claim when one S2/S3 ID is claimed by multiple S1 entities above threshold.
The exact Hungarian-algorithm variant is a documented upgrade path, not
built here: greedy is the Phase 0 MVP."""
from __future__ import annotations

import numpy as np
import polars as pl


def resolve_1to1(candidates: pl.DataFrame, probs: np.ndarray, threshold: float) -> dict[str, set[str]]:
    s1_ids = candidates["source1_entity_id"].to_list()
    cand_ids = candidates["candidate_id"].to_list()

    best_claimant: dict[str, tuple[str, float]] = {}
    for s1_id, cand_id, p in zip(s1_ids, cand_ids, probs):
        if p < threshold:
            continue
        current = best_claimant.get(cand_id)
        if current is None or p > current[1]:
            best_claimant[cand_id] = (s1_id, p)

    result: dict[str, set[str]] = {s1_id: set() for s1_id in set(s1_ids)}
    for cand_id, (s1_id, _p) in best_claimant.items():
        result[s1_id].add(cand_id)
    return result
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_assignment.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/inference/assignment.py code/business_entity_resolution/tests/test_assignment.py
git commit -m "feat: add greedy 1-to-1 assignment resolver (PRD §13)"
```

---

### Task 8: Output writer (§22) with subset invariant

**Files:**
- Create: `code/business_entity_resolution/src/inference/write_outputs.py`
- Test: `code/business_entity_resolution/tests/test_write_outputs.py`

**Interfaces:**
- Consumes: `matches: dict[str, set[str]]` (Task 7's output), `candidate_ids_by_s1: dict[str, set[str]]` (from Task 2's blocking output, pre-threshold), `all_test_s1_ids: list[str]` (from `test_source1`).
- Produces: `write_outputs(all_test_s1_ids, matches, candidate_ids_by_s1, matching_path: Path, candidate_path: Path) -> None`. Enforces `matches[s1_id] <= candidate_ids_by_s1.get(s1_id, set())` by construction: intersects before writing rather than trusting the caller.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_write_outputs.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.write_outputs import write_outputs

def test_every_test_s1_id_gets_exactly_one_row(tmp_path):
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1", "S1-2", "S1-3"],
        matches={"S1-1": {"S2-1"}},
        candidate_ids_by_s1={"S1-1": {"S2-1", "S2-2"}, "S1-2": set()},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    lines = matching_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 4  # header + 3 entities
    body_ids = {ln.split("\t")[0] for ln in lines[1:]}
    assert body_ids == {"S1-1", "S1-2", "S1-3"}

def test_empty_match_serializes_as_nothing_after_tab(tmp_path):
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1"], matches={}, candidate_ids_by_s1={},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    line = matching_path.read_text(encoding="utf-8").splitlines()[1]
    assert line == "S1-1\t"

def test_match_not_in_candidates_is_dropped_not_written(tmp_path):
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    # S2-99 is claimed as a match but was never a candidate -- must not appear.
    write_outputs(
        all_test_s1_ids=["S1-1"], matches={"S1-1": {"S2-99"}},
        candidate_ids_by_s1={"S1-1": {"S2-1"}},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    line = matching_path.read_text(encoding="utf-8").splitlines()[1]
    assert line == "S1-1\t"

def test_subset_invariant_holds_across_full_file(tmp_path):
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1"], matches={"S1-1": {"S2-1"}},
        candidate_ids_by_s1={"S1-1": {"S2-1", "S2-2"}},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    m = {ln.split("\t")[0]: set(ln.split("\t")[1].split(",")) if len(ln.split("\t")) > 1 and ln.split("\t")[1] else set()
         for ln in matching_path.read_text(encoding="utf-8").splitlines()[1:]}
    c = {ln.split("\t")[0]: set(ln.split("\t")[1].split(",")) if len(ln.split("\t")) > 1 and ln.split("\t")[1] else set()
         for ln in candidate_path.read_text(encoding="utf-8").splitlines()[1:]}
    for s1_id, matched in m.items():
        assert matched <= c.get(s1_id, set())
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_write_outputs.py -v` → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference/write_outputs.py
"""matching_results.tsv / candidate_pairs.tsv writer (PRD §22).

Row driver is `all_test_s1_ids`, never a dict's keys, so a silently-dropped
entity can never simply be missing. The subset invariant (matched ids
subset of candidate ids, per entity) is enforced here by intersecting
before writing -- not trusted from the caller.
"""
from __future__ import annotations

from pathlib import Path


def _write_tsv(path: Path, header: str, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for s1_id, joined in rows:
            f.write(f"{s1_id}\t{joined}\n")


def write_outputs(
    all_test_s1_ids: list[str],
    matches: dict[str, set[str]],
    candidate_ids_by_s1: dict[str, set[str]],
    matching_path: Path,
    candidate_path: Path,
) -> None:
    matching_rows = []
    candidate_rows = []
    for s1_id in all_test_s1_ids:
        cands = candidate_ids_by_s1.get(s1_id, set())
        matched = matches.get(s1_id, set()) & cands
        matching_rows.append((s1_id, ",".join(sorted(matched))))
        candidate_rows.append((s1_id, ",".join(sorted(cands))))

    matching_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(matching_path, "source1_entity_id\tmatched_entity_ids", matching_rows)
    _write_tsv(candidate_path, "source1_entity_id\tcandidate_entity_ids", candidate_rows)
```

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_write_outputs.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/inference/write_outputs.py code/business_entity_resolution/tests/test_write_outputs.py
git commit -m "feat: add output writer enforcing subset invariant by construction (PRD §22)"
```

---

### Task 9: `run_pipeline.py` — wire it all together + validator gate

**Files:**
- Create: `code/business_entity_resolution/src/run_pipeline.py`
- Modify: `code/business_entity_resolution/README.md` — update "Current status" section once this lands

**Interfaces:**
- Consumes every module from Tasks 1–8 plus `src.data.loaders.load_all`.
- Produces: a `main()` that, given `data_dir` and `output_dir`, runs the full train→threshold→infer→write→validate sequence and exits non-zero if `validate_submission.py` fails.

- [ ] **Step 1: Write minimal implementation** (no new unit test — this is an integration entry point; Task 10 is its real-data proof)

```python
# src/run_pipeline.py
"""PRD §22 single entry point: raw TSV -> validated output/.

python -m src.run_pipeline [data_dir] [output_dir]
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from .blocking.lexical import generate_candidates
from .data.loaders import BERDataset, default_data_dir, load_all
from .evaluation.metrics import evaluate
from .features.pairwise import build_features
from .inference.assignment import resolve_1to1
from .inference.threshold import sweep_threshold
from .inference.write_outputs import write_outputs
from .models.classifier import predict_proba, train_classifier

SEED = 42


def _label_training_pairs(candidates: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    gt = ground_truth.select(
        "source1_entity_id",
        pl.col("matched_entity_ids").str.split(",").alias("mid"),
    ).explode("mid", empty_as_null=False)
    positive_pairs = set(zip(gt["source1_entity_id"].to_list(), gt["mid"].to_list()))
    is_positive = [
        1 if (s1, cid) in positive_pairs else 0
        for s1, cid in zip(candidates["source1_entity_id"].to_list(), candidates["candidate_id"].to_list())
    ]
    return candidates.with_columns(label=pl.Series(is_positive))


def run(data_dir: Path, output_dir: Path, repo_root: Path) -> int:
    random.seed(SEED)
    np.random.seed(SEED)

    ds: BERDataset = load_all(data_dir)

    print("[pipeline] blocking training candidates...")
    train_candidates = generate_candidates(ds.train_source1, ds.train_source2, ds.train_source3, top_k=20)
    train_candidates = _label_training_pairs(train_candidates, ds.train_ground_truth)

    print("[pipeline] building training features...")
    train_features = build_features(train_candidates, ds.train_source1, ds.train_source2, ds.train_source3)

    print("[pipeline] naive random row split (PRD §19 Phase 0 -- GroupKFold is Phase 2)...")
    train_df, val_df = train_test_split(train_features, test_size=0.2, random_state=SEED, stratify=train_features["label"].to_numpy())

    print("[pipeline] training classifier...")
    model = train_classifier(train_df, seed=SEED)

    val_probs = predict_proba(model, val_df)
    val_truth = {
        s1: set(m.split(",")) if m else set()
        for s1, m in zip(ds.train_ground_truth["source1_entity_id"].to_list(), ds.train_ground_truth["matched_entity_ids"].to_list())
        if s1 in set(val_df["source1_entity_id"].to_list())
    }
    best_threshold, best_val_f05 = sweep_threshold(val_df, val_probs, val_truth)
    print(f"[pipeline] naive-split local macro F0.5 = {best_val_f05:.4f} at threshold {best_threshold}")
    metrics = evaluate(resolve_1to1(val_df, val_probs, best_threshold), val_truth)
    print(f"[pipeline] full val metrics: {metrics}")

    print("[pipeline] blocking test candidates...")
    test_candidates = generate_candidates(ds.test_source1, ds.test_source2, ds.test_source3, top_k=20)
    test_features = build_features(test_candidates, ds.test_source1, ds.test_source2, ds.test_source3)
    test_probs = predict_proba(model, test_features)
    test_matches = resolve_1to1(test_features, test_probs, best_threshold)

    candidate_ids_by_s1: dict[str, set[str]] = {}
    for s1_id, cand_id in zip(test_candidates["source1_entity_id"].to_list(), test_candidates["candidate_id"].to_list()):
        candidate_ids_by_s1.setdefault(s1_id, set()).add(cand_id)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    write_outputs(ds.test_source1["entity_id"].to_list(), test_matches, candidate_ids_by_s1, matching_path, candidate_path)

    print("[pipeline] running validate_submission.py...")
    result = subprocess.run(
        [sys.executable, str(repo_root / "student_resource" / "utils" / "validate_submission.py"),
         "--matching", str(matching_path), "--candidate", str(candidate_path),
         "--test-dir", str(data_dir / "test")],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("[pipeline] validate_submission.py FAILED -- submission not safe to submit")
        return result.returncode
    print("[pipeline] PASS -- validate_submission.py green")
    return 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data_dir()
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else repo_root / "output"
    sys.exit(run(data_dir, output_dir, repo_root))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add code/business_entity_resolution/src/run_pipeline.py
git commit -m "feat: wire Phase 0 pipeline end-to-end with validate_submission.py gate"
```

---

### Task 10: Run it for real, produce the submission, log the number

**Files:** none new — this is the execution/verification task.

- [ ] **Step 1:** `pytest code/business_entity_resolution/tests/ -q` — full unit suite green (fast, synthetic fixtures only).
- [ ] **Step 2:** From `code/business_entity_resolution/`, run `python -m src.run_pipeline` against the real dataset (`../../student_resource/dataset`), writing to `../../output/`. Expect this to take a while at ~2.2M/1.7M S1 scale — run in background, monitor stdout for the per-stage `[pipeline]` print lines.
- [ ] **Step 3:** Confirm exit code 0 (`validate_submission.py` passed) and record the printed naive-split macro F₀.₅ number.
- [ ] **Step 4:** Update `code/business_entity_resolution/README.md`'s "Current status" section to Phase 0 complete, and add the measured F₀.₅ number + threshold to `PRD.md` §25 (Success Metrics) or a new `PHASE0_RESULTS.md`, whichever the existing doc convention prefers.
- [ ] **Step 5:** Commit `output/matching_results.tsv`, `output/candidate_pairs.tsv`? — **no**: check `.gitignore` first; PRD §21 puts `output/` inside the submission ZIP, not necessarily inside the git repo. Decide based on repo's existing `.gitignore` policy (memory: dataset TSVs are excluded; output files are small — likely fine to commit, but confirm size first).
- [ ] **Step 6:** Commit the README/results doc update.

```bash
git add code/business_entity_resolution/README.md
git commit -m "docs: mark Phase 0 complete with measured naive-split F0.5"
```

---

## Self-Review Notes

- **Spec coverage:** §12 exact+TF-IDF (Task 2), §13 MVP features (Task 3), §14 single LightGBM (Task 5), §16 one global threshold (Task 6), §13 1-to-1 (Task 7), §22 writer+subset invariant (Task 8), §23 validator gate + seed (Task 9), §17 metrics (Task 4), §19 Phase 0 exit criteria (Task 10). §10 full normalization, §12 fallback/cross-country buffer, §14 hard-negative mining/singleton classifier/cross-encoder, §15 GroupKFold, §16 per-country calibration are explicitly out of scope (later phases) — confirmed against roadmap wording for each.
- **Type consistency:** `FEATURE_COLUMNS` (Task 3) is the single source of truth consumed identically by Task 5 (`classifier.py`) and Task 9 (`run_pipeline.py` passes full feature frames straight through). `candidates` schema (`source1_entity_id, candidate_id, country, blocking_rank, tfidf_score, exact_match`) from Task 2 flows unchanged through Tasks 3, 6, 7.
- **Placeholder scan:** none found — every step has real code.
- **Known open risk (flagged, not resolved in-plan):** Jaro-Winkler stand-in in Task 3 needs a one-line verification against the pinned `rapidfuzz==3.14.6` API before the "done" commit — call this out during execution rather than guessing.
