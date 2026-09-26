"""Unit tests for src.inference.threshold -- global-threshold sweep."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.inference.threshold import sweep_threshold  # noqa: E402


def test_sweep_finds_threshold_that_separates_classes() -> None:
    candidates = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
        "candidate_id": ["S2-1", "S2-2", "S2-3"],
    })
    probs = np.array([0.9, 0.1, 0.05])
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    best_t, best_score = sweep_threshold(candidates, probs, truth, grid=[0.2, 0.5, 0.8])
    assert best_score == pytest.approx(1.0)
    assert 0.2 <= best_t <= 0.8
