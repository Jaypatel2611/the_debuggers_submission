"""Unit tests for src.inference.assignment -- greedy 1-to-1 resolver."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.inference.assignment import resolve_1to1  # noqa: E402


def test_contested_id_goes_to_highest_probability_claimant() -> None:
    candidates = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "candidate_id": ["S2-1", "S2-1"],
    })
    probs = np.array([0.6, 0.9])
    result = resolve_1to1(candidates, probs, threshold=0.5)
    assert result["S1-2"] == {"S2-1"}
    assert result["S1-1"] == set()


def test_below_threshold_never_claimed() -> None:
    candidates = pl.DataFrame({"source1_entity_id": ["S1-1"], "candidate_id": ["S2-1"]})
    probs = np.array([0.1])
    result = resolve_1to1(candidates, probs, threshold=0.5)
    assert result["S1-1"] == set()


def test_uncontested_ids_all_kept() -> None:
    candidates = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "candidate_id": ["S2-1", "S2-2"],
    })
    probs = np.array([0.9, 0.8])
    result = resolve_1to1(candidates, probs, threshold=0.5)
    assert result["S1-1"] == {"S2-1", "S2-2"}
