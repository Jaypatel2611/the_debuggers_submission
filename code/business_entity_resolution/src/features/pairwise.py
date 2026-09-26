"""Pairwise similarity features -- PRD §13 MVP subset for Phase 0.

Excluded here on purpose (later phases, not missing by accident):
legal-suffix match flag (needs §10's suffix table, Phase 3), PIN/postal
exact-match + city/state flags (need address-component parsing, Phase 3),
bi-encoder cosine + cross-encoder score (Phase 5/6). Candidate rank and
TF-IDF score are kept -- both already produced by blocking, free here.
"""
from __future__ import annotations

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from ..preprocessing.normalize import normalize_text

FEATURE_COLUMNS = [
    "lev_ratio", "jaro_winkler", "token_sort_ratio", "token_set_ratio",
    "char_trigram_jaccard", "name_len_delta", "country_match",
    "addr_token_jaccard", "blocking_rank", "tfidf_score",
]

_EMPTY_FEATURE_SCHEMA = {
    "lev_ratio": pl.Float64, "jaro_winkler": pl.Float64, "token_sort_ratio": pl.Float64,
    "token_set_ratio": pl.Float64, "char_trigram_jaccard": pl.Float64, "name_len_delta": pl.Int64,
    "country_match": pl.Int64, "addr_token_jaccard": pl.Float64,
}


def _trigrams(s: str) -> set[str]:
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


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
        return candidates.with_columns(**{c: pl.Series([], dtype=t) for c, t in _EMPTY_FEATURE_SCHEMA.items()})

    s1_lookup = s1.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("business_name").alias("name_a"),
        pl.col("business_address").alias("addr_a"),
        pl.col("country").alias("country_a"),
    )
    cand_lookup = pl.concat([s2, s3], how="vertical").select(
        pl.col("entity_id").alias("candidate_id"),
        pl.col("business_name").alias("name_b"),
        pl.col("business_address").alias("addr_b"),
        pl.col("country").alias("country_b"),
    )

    joined = candidates.join(s1_lookup, on="source1_entity_id", how="left").join(
        cand_lookup, on="candidate_id", how="left"
    )

    names_a = [normalize_text(n) for n in joined["name_a"].to_list()]
    names_b = [normalize_text(n) for n in joined["name_b"].to_list()]
    addrs_a = [normalize_text(a) for a in joined["addr_a"].to_list()]
    addrs_b = [normalize_text(a) for a in joined["addr_b"].to_list()]
    countries_a = joined["country_a"].to_list()
    countries_b = joined["country_b"].to_list()

    n = joined.height
    lev_ratio = [0.0] * n
    jaro_winkler = [0.0] * n
    token_sort_ratio = [0.0] * n
    token_set_ratio = [0.0] * n
    char_trigram_jaccard = [0.0] * n
    name_len_delta = [0] * n
    country_match = [0] * n
    addr_token_jaccard = [0.0] * n

    for i in range(n):
        na, nb = names_a[i], names_b[i]
        lev_ratio[i] = fuzz.ratio(na, nb) / 100.0
        jaro_winkler[i] = JaroWinkler.normalized_similarity(na, nb)
        token_sort_ratio[i] = fuzz.token_sort_ratio(na, nb)
        token_set_ratio[i] = fuzz.token_set_ratio(na, nb)
        char_trigram_jaccard[i] = _trigram_jaccard(na, nb)
        name_len_delta[i] = abs(len(na) - len(nb))
        country_match[i] = int(countries_a[i] == countries_b[i])
        addr_token_jaccard[i] = _token_jaccard(addrs_a[i], addrs_b[i])

    return joined.drop(["name_a", "addr_a", "country_a", "name_b", "addr_b", "country_b"]).with_columns(
        lev_ratio=pl.Series(lev_ratio),
        jaro_winkler=pl.Series(jaro_winkler),
        token_sort_ratio=pl.Series(token_sort_ratio),
        token_set_ratio=pl.Series(token_set_ratio),
        char_trigram_jaccard=pl.Series(char_trigram_jaccard),
        name_len_delta=pl.Series(name_len_delta),
        country_match=pl.Series(country_match),
        addr_token_jaccard=pl.Series(addr_token_jaccard),
    )
