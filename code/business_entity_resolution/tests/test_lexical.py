"""Unit tests for src.blocking.lexical -- exact+TF-IDF per-country blocking."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.blocking.lexical import generate_candidates  # noqa: E402


def _df(rows: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


def test_exact_match_always_included() -> None:
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "US"), ("S2-2", "Totally Different Biz", "9 Elm St", "US")])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=1)
    ids = result.filter(pl.col("source1_entity_id") == "S1-1")["candidate_id"].to_list()
    assert "S2-1" in ids


def test_tfidf_ranks_similar_name_above_dissimilar() -> None:
    s1 = _df([("S1-1", "Acme Corporation", "1 Main St", "US")])
    s2 = _df([
        ("S2-1", "Acme Corp", "1 Main St", "US"),
        ("S2-2", "Zzyzx Unrelated Widgets", "9 Elm St", "US"),
    ])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=2).sort("blocking_rank")
    top = result.filter(pl.col("source1_entity_id") == "S1-1").row(0, named=True)
    assert top["candidate_id"] == "S2-1"


def test_never_crosses_country() -> None:
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "India")])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=5)
    assert result.filter(pl.col("source1_entity_id") == "S1-1").height == 0


def test_no_same_country_candidates_returns_empty_not_crash() -> None:
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "France")])
    s2 = _df([("S2-1", "Acme Corp", "1 Main St", "US")])
    s3 = _df([])
    result = generate_candidates(s1, s2, s3, top_k=5)
    assert result.height == 0
