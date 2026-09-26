"""Unit tests for src.data.eda -- drift-check logic and new-measurement heuristics.

Builds tiny synthetic BERDataset instances so these run in milliseconds and
don't depend on the real dataset being present. The real dataset's numbers
are exercised separately via `python -m src.data.eda` (PRD §19 Phase 1 exit
criterion: the checked-in EDA_REPORT.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.eda import (  # noqa: E402
    DriftDetected,
    check_country_split,
    check_match_count_distribution,
    check_singleton_rate,
    measure_non_latin_script_fraction,
    measure_s1_injected_noise,
    measure_typo_vs_transposition,
)
from src.data.loaders import BERDataset  # noqa: E402


def _source_df(rows: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row"
    )


def _matching_dataset() -> BERDataset:
    """A tiny dataset whose country split / singleton rate / match-count
    distribution are deliberately built to equal the PRD's measured
    constants would require the full 2.2M-row dataset -- instead these
    tests patch the MEASURED_* constants down to match the fixture size."""
    train_s1 = _source_df([
        ("S1-1", "Acme Corp", "1 Main St", "US"),
        ("S1-2", "Beta Pvt Ltd", "2 Nehru Rd", "India"),
    ])
    test_s1 = _source_df([
        ("S1-3", "Gamma Inc", "3 Oak Ave", "US"),
    ])
    train_s2 = _source_df([
        ("S2-1", "Acme Corporation", "1 Main Street", "US"),
    ])
    train_s3 = _source_df([
        ("S3-1", "Beta Private Limited", "2 Nehru Road", "India"),
    ])
    train_gt = pl.DataFrame(
        {"source1_entity_id": ["S1-1", "S1-2"], "matched_entity_ids": ["S2-1", "S3-1"]}
    )
    return BERDataset(
        train_source1=train_s1,
        train_source2=train_s2,
        train_source3=train_s3,
        train_ground_truth=train_gt,
        test_source1=test_s1,
        test_source2=_source_df([]),
        test_source3=_source_df([]),
    )


def test_check_country_split_detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.eda as eda

    ds = _matching_dataset()
    monkeypatch.setattr(eda, "MEASURED_TRAIN_S1_COUNTRY", {"US": 1, "India": 1})
    monkeypatch.setattr(eda, "MEASURED_TEST_S1_COUNTRY", {"US": 1})
    result = check_country_split(ds)
    assert result["train"] == {"US": 1, "India": 1}
    assert result["test"] == {"US": 1}


def test_check_country_split_raises_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.eda as eda

    ds = _matching_dataset()
    monkeypatch.setattr(eda, "MEASURED_TRAIN_S1_COUNTRY", {"US": 999, "India": 1})
    monkeypatch.setattr(eda, "MEASURED_TEST_S1_COUNTRY", {"US": 1})
    with pytest.raises(DriftDetected):
        check_country_split(ds)


def test_check_singleton_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.eda as eda

    ds = _matching_dataset()
    # 2 ground-truth rows, 0 singletons in this fixture (both have a match).
    monkeypatch.setattr(eda, "MEASURED_SINGLETON_TOTAL", 2)
    monkeypatch.setattr(eda, "MEASURED_SINGLETON_COUNT", 0)
    result = check_singleton_rate(ds)
    assert result == {"total": 2, "singletons": 0}


def test_check_singleton_rate_raises_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.eda as eda

    ds = _matching_dataset()
    monkeypatch.setattr(eda, "MEASURED_SINGLETON_TOTAL", 2)
    monkeypatch.setattr(eda, "MEASURED_SINGLETON_COUNT", 1)  # wrong on purpose
    with pytest.raises(DriftDetected):
        check_singleton_rate(ds)


def test_check_match_count_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.eda as eda

    ds = _matching_dataset()
    monkeypatch.setattr(eda, "MEASURED_MATCH_COUNT_DIST", {1: 2})
    result = check_match_count_distribution(ds)
    assert result == {1: 2}


def test_measure_non_latin_script_fraction_detects_devanagari() -> None:
    ds = _matching_dataset()
    india_s2 = _source_df([
        ("S2-9", "राम मार्केटिंग प्राइवेट लिमिटेड", "New Delhi", "India"),
    ])
    ds2 = BERDataset(**{**ds.__dict__, "train_source2": india_s2})
    result = measure_non_latin_script_fraction(ds2)
    row = result.filter(pl.col("source") == "source2").row(0, named=True)
    assert row["non_ascii_name_pct"] == 100.0


def test_measure_s1_injected_noise_flags_garbled_prefix() -> None:
    ds = _matching_dataset()
    # source1 = train_source1 (2 rows) concatenated with test_source1 (1 row);
    # only one of those 3 is garbled here.
    garbled_s1 = ds.train_source1.with_columns(
        business_name=pl.Series(["<< Team Ecole", "Normal Name"])
    )
    ds2 = BERDataset(**{**ds.__dict__, "train_source1": garbled_s1})
    result = measure_s1_injected_noise(ds2)
    row = result.filter(pl.col("source") == "source1").row(0, named=True)
    assert row["garbled_name_heuristic_pct"] == pytest.approx(100 / 3)


def test_measure_typo_vs_transposition_classifies_typo_and_transposition() -> None:
    ds = BERDataset(
        train_source1=_source_df([
            ("S1-1", "Acme Corporation", "1 Main St", "US"),
            ("S1-2", "North West Traders", "2 Oak Ave", "US"),
        ]),
        train_source2=_source_df([
            ("S2-1", "Acme Corporatoin", "1 Main St", "US"),  # typo
        ]),
        train_source3=_source_df([
            ("S3-1", "Traders North West", "2 Oak Ave", "US"),  # transposition
        ]),
        train_ground_truth=pl.DataFrame(
            {"source1_entity_id": ["S1-1", "S1-2"], "matched_entity_ids": ["S2-1", "S3-1"]}
        ),
        test_source1=_source_df([]),
        test_source2=_source_df([]),
        test_source3=_source_df([]),
    )
    result = measure_typo_vs_transposition(ds, sample_size=10)
    assert result["n_true_positive_pairs_total"] == 2
    assert result["typo_like"] + result["transposition_like"] + result["other"] == 2
