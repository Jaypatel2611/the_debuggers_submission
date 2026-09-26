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
