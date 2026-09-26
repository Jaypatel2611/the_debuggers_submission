# Business Entity Resolution — Pipeline Code

Team **the_debuggers** · Amazon ML Challenge 2026. Implements the pipeline
described in `../../ENTITY_RESOLUTION_PRD.md` (also mirrored as `PRD.md` at
the repo root). This README documents what is **actually implemented**,
not the full roadmap — see the PRD §19 for the phase-by-phase plan.

## Current status: PRD §19 Phase 0 (loaders) + Phase 1 (EDA)

Implemented so far:

- `src/data/loaders.py` — TSV loaders for all six source files plus
  `train_ground_truth.tsv`, with hard row-count and header-schema
  validation (PRD §2).
- `src/data/eda.py` — the PRD §9 noise-measurement spec: drift checks
  against the PRD's already-measured numbers (country split, singleton
  rate, match-count distribution), plus four new measurements (address
  completeness, non-Latin-script fraction, S1 noise-injection rate,
  typo-vs-transposition frequency).

Everything else in the PRD's folder layout (§21) exists only as empty,
scaffolded packages (`preprocessing/`, `blocking/`, `features/`, `models/`,
`validation/`, `inference/`, `evaluation/`) — no modeling, preprocessing, or
blocking code has been written yet. Do not assume those directories do
anything.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Reproduce what exists today

Both commands below are run from this directory (`code/business_entity_resolution/`)
and expect the challenge's dataset at `../../student_resource/dataset`
(the default; override by passing a path as the first argument).

```bash
# Load + row-count-validate all 7 files against the real dataset.
python -m src.data.loaders

# Regenerate the checked-in noise-distribution report at the repo root
# (EDA_REPORT.md) -- PRD §19 Phase 1's exit criterion.
python -m src.data.eda
```

Run the unit test suite (synthetic fixtures, no dataset required, <1s):

```bash
pytest tests/ -q
```

## Notable implementation details worth knowing before touching this code

- **Row-count constants in `loaders.py` are data-row counts, not `wc -l`
  output.** The PRD's §2 table was originally populated with `wc -l`
  (which counts the header line); `pl.read_csv` returns data rows only, so
  every constant is the PRD number minus 1. See `CHANGELOG.md`.
- **No lemmatization anywhere, ever** — PRD §10 is a hard constraint, not a
  style preference. If you're adding `preprocessing/`, read that section
  first.
- **No pandas.** Polars only, per PRD §20 — including in report generation
  (`eda.py` has its own tiny Markdown table renderer instead of
  `df.to_pandas().to_markdown()` to avoid pulling in a pandas+tabulate
  dependency for one report).
- **State-name matching in `eda.py`'s address-completeness heuristic covers
  US (abbreviations + full names) and Indian states only.** France's
  ~99% "missing state" rate in `EDA_REPORT.md` is this heuristic's known
  blind spot (French régions/départements aren't in the list), not a real
  data-completeness finding — don't cite it as one.
