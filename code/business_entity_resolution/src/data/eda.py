"""Noise-distribution EDA for Business Entity Resolution (PRD §9).

Two kinds of function live here:

* ``check_*`` -- recomputes a number PRD §9 *already measured* directly
  against the shipped dataset, and raises ``DriftDetected`` if the freshly
  computed value disagrees. These are drift guards, not new findings --
  every later phase cites the PRD's numbers, so if the dataset on disk ever
  stops matching them, that has to be loud, not silent.
* ``measure_*`` -- the PRD §9 "what to measure" items that were *not* yet
  measured when the PRD was written. These just report numbers.

No modeling, preprocessing, or blocking code belongs in this file (PRD §19
Phase 1 scope) -- only measurement.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from .loaders import BERDataset, default_data_dir, load_all

# ---------------------------------------------------------------------------
# PRD §9 / §1 -- numbers already measured directly against the dataset.
# ---------------------------------------------------------------------------

MEASURED_TRAIN_S1_COUNTRY = {"US": 1_323_633, "India": 883_188}
MEASURED_TEST_S1_COUNTRY = {"US": 663_106, "India": 809_986, "France": 259_452}
MEASURED_SINGLETON_COUNT = 123_247
MEASURED_SINGLETON_TOTAL = 2_206_821  # train_ground_truth data rows (header excluded)
MEASURED_MATCH_COUNT_DIST = {
    1: 119_157, 2: 375_212, 3: 530_841, 4: 484_115, 5: 321_957,
    6: 164_868, 7: 63_968, 8: 18_680, 9: 4_205, 10: 534, 11: 37,
}


class DriftDetected(AssertionError):
    """A freshly computed PRD §9 number no longer matches what PRD §9 reports."""


# ---------------------------------------------------------------------------
# check_* -- drift guards against PRD §9's already-measured numbers.
# ---------------------------------------------------------------------------

def check_country_split(ds: BERDataset) -> dict[str, dict[str, int]]:
    train_counts = dict(
        ds.train_source1.group_by("country").len().iter_rows()
    )
    test_counts = dict(
        ds.test_source1.group_by("country").len().iter_rows()
    )
    if train_counts != MEASURED_TRAIN_S1_COUNTRY:
        raise DriftDetected(
            f"train_source1 country split changed: {train_counts} != "
            f"{MEASURED_TRAIN_S1_COUNTRY} (PRD §9)"
        )
    if test_counts != MEASURED_TEST_S1_COUNTRY:
        raise DriftDetected(
            f"test_source1 country split changed: {test_counts} != "
            f"{MEASURED_TEST_S1_COUNTRY} (PRD §9)"
        )
    return {"train": train_counts, "test": test_counts}


def check_singleton_rate(ds: BERDataset) -> dict[str, int]:
    total = ds.train_ground_truth.height
    singleton_count = ds.train_ground_truth.filter(
        pl.col("matched_entity_ids") == ""
    ).height
    if total != MEASURED_SINGLETON_TOTAL:
        raise DriftDetected(
            f"train_ground_truth row count changed: {total} != "
            f"{MEASURED_SINGLETON_TOTAL} (PRD §9)"
        )
    if singleton_count != MEASURED_SINGLETON_COUNT:
        raise DriftDetected(
            f"singleton count changed: {singleton_count} != "
            f"{MEASURED_SINGLETON_COUNT} (PRD §1/§9)"
        )
    return {"total": total, "singletons": singleton_count}


def check_match_count_distribution(ds: BERDataset) -> dict[int, int]:
    counts = (
        ds.train_ground_truth
        .filter(pl.col("matched_entity_ids") != "")
        .select(pl.col("matched_entity_ids").str.split(",").list.len().alias("n"))
        .group_by("n")
        .len()
        .sort("n")
    )
    dist = dict(counts.iter_rows())
    # PRD §9 buckets 7..11 together as "7+"; compare bucket-for-bucket here
    # instead of collapsing, since collapsing would hide a shift within the
    # tail (e.g. 9 growing while 11 shrinks) that a single "7+" sum can't see.
    if dist != MEASURED_MATCH_COUNT_DIST:
        raise DriftDetected(
            f"match-count distribution changed: {dist} != "
            f"{MEASURED_MATCH_COUNT_DIST} (PRD §9)"
        )
    return dist


# ---------------------------------------------------------------------------
# measure_* -- PRD §9 items not yet measured. New findings, nothing to
# assert against -- these become the next PRD revision's [FACT] entries.
# ---------------------------------------------------------------------------

_POSTAL_RE = re.compile(r"\d{4,6}")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_GARBLED_RE = re.compile(r"^[^A-Za-z0-9]|<<|>>")

# ponytail: static, hand-written subdivision lists for a diagnostic
# "does this address mention a state/region" heuristic only -- not a
# gazetteer, not used for matching, not looked up externally. Coarser
# region coverage (more countries) is the upgrade path if this EDA metric
# needs to generalize beyond US/India.
_US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY",
}
# Full names too -- caught by actually running this against the real data:
# source3's US addresses spell the state out ("Fort Worth, Texas") rather
# than abbreviating it, and abbreviations alone put missing-state at an
# implausible 95%. See CHANGELOG.md.
_US_STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa",
    "kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new hampshire","new jersey","new mexico","new york","north carolina",
    "north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming",
}
_INDIAN_STATES = {
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa",
    "gujarat","haryana","himachal pradesh","jharkhand","karnataka","kerala",
    "madhya pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
    "odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura",
    "uttar pradesh","uttarakhand","west bengal","delhi",
}
_STATE_NAME_ALTERNATION = "|".join(
    sorted({s.title() for s in _US_STATE_NAMES | _INDIAN_STATES}, key=len, reverse=True)
)
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted(_US_STATE_ABBR)) + "|" + _STATE_NAME_ALTERNATION + r")\b",
    re.IGNORECASE,
)


def measure_address_completeness(ds: BERDataset) -> pl.DataFrame:
    """% blank / missing-postal-heuristic / missing-state-heuristic, per source x country."""
    frames = {
        "source1": pl.concat([ds.train_source1, ds.test_source1]),
        "source2": pl.concat([ds.train_source2, ds.test_source2]),
        "source3": pl.concat([ds.train_source3, ds.test_source3]),
    }
    rows = []
    for source_name, df in frames.items():
        stats = (
            df.group_by("country")
            .agg(
                n=pl.len(),
                blank_address_pct=(pl.col("business_address") == "").mean() * 100,
                missing_postal_heuristic_pct=(
                    ~pl.col("business_address").str.contains(_POSTAL_RE.pattern)
                ).mean() * 100,
                missing_state_heuristic_pct=(
                    ~pl.col("business_address").str.contains(_STATE_RE.pattern)
                ).mean() * 100,
            )
            .with_columns(source=pl.lit(source_name))
        )
        rows.append(stats)
    return pl.concat(rows).select(
        "source", "country", "n", "blank_address_pct",
        "missing_postal_heuristic_pct", "missing_state_heuristic_pct",
    ).sort(["source", "country"])


def measure_non_latin_script_fraction(ds: BERDataset) -> pl.DataFrame:
    """% of India-labeled business_name values containing a non-ASCII character."""
    frames = {
        "source1": pl.concat([ds.train_source1, ds.test_source1]),
        "source2": pl.concat([ds.train_source2, ds.test_source2]),
        "source3": pl.concat([ds.train_source3, ds.test_source3]),
    }
    rows = []
    for source_name, df in frames.items():
        india = df.filter(pl.col("country") == "India")
        pct = (
            india.select(
                (pl.col("business_name").str.contains(_NON_ASCII_RE.pattern)).mean() * 100
            ).item()
            if india.height
            else None
        )
        rows.append({"source": source_name, "n_india": india.height, "non_ascii_name_pct": pct})
    return pl.DataFrame(rows)


def measure_s1_injected_noise(ds: BERDataset) -> pl.DataFrame:
    """Garbled-name heuristic rate in S1 vs. S2/S3, as a reference comparison."""
    frames = {
        "source1": pl.concat([ds.train_source1, ds.test_source1]),
        "source2": pl.concat([ds.train_source2, ds.test_source2]),
        "source3": pl.concat([ds.train_source3, ds.test_source3]),
    }
    rows = []
    for source_name, df in frames.items():
        pct = df.select(
            (pl.col("business_name").str.contains(_GARBLED_RE.pattern)).mean() * 100
        ).item()
        rows.append({"source": source_name, "garbled_name_heuristic_pct": pct})
    return pl.DataFrame(rows)


# Thresholds for the typo-vs-transposition classifier below. Deliberately
# simple cut points on RapidFuzz's 0-100 scale, not tuned against labels --
# document the values themselves so a reader can judge the buckets, not just
# trust the labels.
_TOKEN_SET_HIGH = 85.0
_LEV_HIGH = 85.0
_LEV_LOW = 70.0


def measure_typo_vs_transposition(
    ds: BERDataset, sample_size: int = 50_000, seed: int = 42
) -> dict[str, int | float]:
    """Levenshtein-ratio vs. token-set-ratio on sampled true-positive name pairs.

    ponytail: samples `sample_size` pairs rather than scoring the full
    ~7M true-positive pairs -- RapidFuzz is fast but not vectorized through
    polars, so a full run is a python-level loop. Upgrade path if the exact
    population count matters: chunk the full join through
    `rapidfuzz.process.cdist` instead of sampling.
    """
    gt = ds.train_ground_truth.filter(pl.col("matched_entity_ids") != "")
    pairs = gt.select(
        "source1_entity_id",
        pl.col("matched_entity_ids").str.split(",").alias("mid"),
    ).explode("mid", empty_as_null=False)

    s1_names = ds.train_source1.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("business_name").alias("name_a"),
    )
    s2_names = ds.train_source2.select(
        pl.col("entity_id").alias("mid"), pl.col("business_name").alias("name_b")
    )
    s3_names = ds.train_source3.select(
        pl.col("entity_id").alias("mid"), pl.col("business_name").alias("name_b")
    )

    pairs = pairs.join(s1_names, on="source1_entity_id", how="inner")
    s2_pairs = pairs.filter(pl.col("mid").str.starts_with("S2-")).join(
        s2_names, on="mid", how="inner"
    )
    s3_pairs = pairs.filter(pl.col("mid").str.starts_with("S3-")).join(
        s3_names, on="mid", how="inner"
    )
    all_pairs = pl.concat([s2_pairs.select("name_a", "name_b"), s3_pairs.select("name_a", "name_b")])

    n_available = all_pairs.height
    n = min(sample_size, n_available)
    sample = all_pairs.sample(n=n, seed=seed) if n_available else all_pairs

    typo = transposition = other = 0
    for name_a, name_b in sample.iter_rows():
        lev = fuzz.ratio(name_a, name_b)
        tset = fuzz.token_set_ratio(name_a, name_b)
        if tset >= _TOKEN_SET_HIGH and lev >= _LEV_HIGH:
            typo += 1
        elif tset >= _TOKEN_SET_HIGH and lev < _LEV_LOW:
            transposition += 1
        else:
            other += 1

    return {
        "n_true_positive_pairs_total": n_available,
        "n_sampled": n,
        "typo_like": typo,
        "transposition_like": transposition,
        "other": other,
        "typo_like_pct": 100 * typo / n if n else 0.0,
        "transposition_like_pct": 100 * transposition / n if n else 0.0,
        "other_pct": 100 * other / n if n else 0.0,
        "thresholds": {"token_set_high": _TOKEN_SET_HIGH, "lev_high": _LEV_HIGH, "lev_low": _LEV_LOW},
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pl.DataFrame) -> str:
    """Minimal Markdown table renderer -- avoids pulling in pandas+tabulate
    just for report formatting (PRD §20 picks Polars over pandas)."""
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(
            f"{v:.2f}" if isinstance(v, float) else str(v) for v in row
        ) + " |"
        for row in df.iter_rows()
    ]
    return "\n".join([header, sep, *body])


def generate_report(data_dir: Path, report_path: Path) -> Path:
    ds = load_all(data_dir)
    lines = ["# EDA Noise-Distribution Report (PRD §9)", ""]

    lines.append("## Drift checks against PRD §9's measured numbers\n")
    for label, fn in (
        ("Country split", check_country_split),
        ("Singleton rate", check_singleton_rate),
        ("Match-count distribution", check_match_count_distribution),
    ):
        try:
            result = fn(ds)
            lines.append(f"- **{label}**: PASS -- matches PRD §9. `{result}`")
        except DriftDetected as exc:
            lines.append(f"- **{label}**: **DRIFT DETECTED** -- {exc}")

    lines.append("\n## New measurements (not previously in the PRD)\n")

    lines.append("### Address-component completeness by source x country\n")
    lines.append(_df_to_markdown(measure_address_completeness(ds)))

    lines.append("\n### Non-Latin-script fraction, India-labeled business_name\n")
    lines.append(_df_to_markdown(measure_non_latin_script_fraction(ds)))

    lines.append("\n### S1 vs. S2/S3 garbled-name heuristic rate\n")
    lines.append(_df_to_markdown(measure_s1_injected_noise(ds)))

    lines.append("\n### Typo vs. word-order-transposition frequency (sampled true positives)\n")
    typo_stats = measure_typo_vs_transposition(ds)
    for k, v in typo_stats.items():
        lines.append(f"- **{k}**: {v}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _demo() -> None:
    """Self-check: run the full report against the real dataset if present."""
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data_dir()
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[4] / "EDA_REPORT.md"
    if not data_dir.is_dir():
        print(f"[eda] dataset dir not found at {data_dir} -- skipping live check")
        return
    path = generate_report(data_dir, out_path)
    print(f"PASS -- report written to {path}")


if __name__ == "__main__":
    _demo()
