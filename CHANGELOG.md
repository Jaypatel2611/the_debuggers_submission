# Changelog

Tracks noticeable, load-bearing changes across the whole repo — not a
commit-by-commit log. Each entry says what changed, why, and which PRD
section it maps to. Newest first.

## 2026-09-26 — Phase 0 + Phase 1: data loaders and EDA

**Scaffolded the PRD §21 folder structure.**
`scripts/scaffold.sh` creates `code/business_entity_resolution/src/{data,
preprocessing,blocking,features,models,validation,inference,evaluation}/`,
`tests/`, `README.md`, `requirements.txt`, and `output/` at the repo root.
Idempotent, re-runnable. Only `src/data/` has real code in it so far —
every other package is an empty scaffold, not a stub implementation.

**Added `src/data/loaders.py`** — Polars-based TSV loaders for all six
source files plus `train_ground_truth.tsv`. Enforces `separator="\t"`
explicitly (the problem statement's documented #1 mistake is a
comma-separated file silently misread), validates the header against the
expected schema, and hard-raises (`RowCountMismatch`, `SchemaMismatch`) on
any deviation rather than warning.

**Bug caught by actually running it against the real dataset, not by
inspection:** the PRD §2 row-count table was populated from `wc -l`, which
counts the header line as a row. `pl.read_csv` returns data-row counts only
(it consumes the header as column names). Every one of the seven expected
row counts was off by exactly 1 as a result — `loaders.py` failed loudly on
the very first real run (`train_source1: expected 2,206,822 rows, got
2,206,821`), which is exactly what the hard-fail-not-warn design was for.
Fixed `EXPECTED_ROWS` to the corrected data-row counts:

| File | PRD §2 table (`wc -l`) | Actual data rows |
|---|---|---|
| train_source1 | 2,206,822 | 2,206,821 |
| train_source2 | 5,034,617 | 5,034,616 |
| train_source3 | 5,285,604 | 5,285,603 |
| train_ground_truth | 2,206,822 | 2,206,821 |
| test_source1 | 1,732,545 | 1,732,544 |
| test_source2 | 4,887,274 | 4,887,273 |
| test_source3 | 5,082,317 | 5,082,316 |

This does not change any of the PRD's percentage-based findings (singleton
rate, country split, match-count distribution) — those were already
computed correctly off data rows via `awk 'NR>1'`. It only affects the
loader's own validation constant. **The PRD's §2 table itself still shows
the `wc -l` figures and should be corrected to data-row counts** the next
time that document is revised, with a footnote explaining the discrepancy.

**Added `src/data/eda.py`** — implements the full PRD §9 measurement spec:

- Three drift-check functions (`check_country_split`,
  `check_singleton_rate`, `check_match_count_distribution`) that recompute
  PRD §9's already-measured numbers from the live dataset and raise
  `DriftDetected` on any disagreement, rather than silently trusting a
  document that could go stale. All three **passed** against the real
  dataset on first full run.
- Four new measurements not previously in the PRD: address-component
  completeness by source × country, non-Latin-script fraction in
  India-labeled names, an S1-vs-S2/S3 garbled-name heuristic (checking
  whether Source 1 itself carries injected noise, not just S2/S3), and a
  sampled typo-vs-word-order-transposition classifier on true-positive
  name pairs (RapidFuzz `ratio` vs. `token_set_ratio`).
- `generate_report()` writes all of the above to `EDA_REPORT.md` at the
  repo root — the PRD §19 Phase 1 exit criterion ("a noise-distribution
  report checked into the repo that every later phase can cite").

**Bug caught the same way, by running against real data instead of trusting
the heuristic in isolation:** the first version of the missing-state
heuristic only matched 2-letter US state abbreviations (`TX`, `OH`, ...).
Against the real data it reported 95.5% of `source3`/US addresses missing a
state, which contradicted the raw sample rows already quoted in the PRD's
own EDA section (`"...Fort Worth, Texas"`, `"...Cincinnati, Ohio"` — full
state names, not abbreviations). Root cause: `source3` addresses
predominantly spell the state out; the abbreviation-only regex never had a
chance to match them. Fixed by adding a 50-state full-name alternation
alongside the abbreviations. Post-fix, `source3`/US missing-state dropped
to 3.29%, in line with `source2`/US's 3.89% — the two sources now agree,
which is the sanity check that confirms the fix rather than just silencing
the symptom. France's address completeness numbers stay near-100%
missing-state after the fix; that remains a real, documented heuristic
blind spot (French régions/départements aren't in the US/India name list),
not a data quality finding — flagged explicitly in
`code/business_entity_resolution/README.md` so it isn't mistaken for one
later.

**Verification performed** (not claimed, actually run):
`pytest tests/ -q` → 14/14 passing (synthetic fixtures, <1s).
`python -m src.data.loaders` → all 7 files load and row-count-validate
against the real ~15M-row dataset in ~4.5s.
`python -m src.data.eda` → full report regenerated against the real
dataset in ~13-16s; all three drift checks PASS.

**Dependencies added:** `polars==1.44.2`, `rapidfuzz==3.14.6`,
`pytest==9.0.3` — pinned in `code/business_entity_resolution/requirements.txt`
per PRD §23.

**Not done yet** (explicitly out of scope for this change, per PRD §19):
no preprocessing, no lemmatization-removal work, no blocking, no feature
engineering, no model training. The scaffolded packages for those stages
are empty on purpose.
