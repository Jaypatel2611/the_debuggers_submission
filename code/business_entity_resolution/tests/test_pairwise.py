"""Unit tests for src.features.pairwise -- §13 MVP feature set."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.features.pairwise import FEATURE_COLUMNS, build_features  # noqa: E402


_SOURCE_SCHEMA = {"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}


def _df(rows: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_SOURCE_SCHEMA)
    return pl.DataFrame(rows, schema=list(_SOURCE_SCHEMA), orient="row")


def test_identical_pair_scores_max_similarity() -> None:
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


def test_country_mismatch_flagged() -> None:
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s3 = _df([("S3-1", "Acme Corp", "1 Main St", "India")])
    s2 = _df([])
    cands = pl.DataFrame({
        "source1_entity_id": ["S1-1"], "candidate_id": ["S3-1"], "country": ["US"],
        "blocking_rank": [0], "tfidf_score": [0.5], "exact_match": [False],
    })
    feats = build_features(cands, s1, s2, s3)
    assert feats.row(0, named=True)["country_match"] == 0


def test_feature_columns_are_all_numeric() -> None:
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


def test_empty_candidates_returns_empty_with_feature_columns() -> None:
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US")])
    s2 = _df([])
    s3 = _df([])
    cands = pl.DataFrame(schema={
        "source1_entity_id": pl.Utf8, "candidate_id": pl.Utf8, "country": pl.Utf8,
        "blocking_rank": pl.Int64, "tfidf_score": pl.Float64, "exact_match": pl.Boolean,
    })
    feats = build_features(cands, s1, s2, s3)
    assert feats.height == 0
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
