"""Single global threshold, swept against exact macro F0.5 (PRD §16) --
per-country thresholding is Phase 7, not here."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..evaluation.metrics import macro_f05


def _predictions_at(candidates: pl.DataFrame, probs: np.ndarray, t: float) -> dict[str, set[str]]:
    keep = probs >= t
    preds: dict[str, set[str]] = {s1_id: set() for s1_id in candidates["source1_entity_id"].unique().to_list()}
    for s1_id, cand_id, k in zip(candidates["source1_entity_id"].to_list(), candidates["candidate_id"].to_list(), keep):
        if k:
            preds[s1_id].add(cand_id)
    return preds


def sweep_threshold(
    candidates: pl.DataFrame, probs: np.ndarray, truth: dict[str, set[str]], grid: list[float] | None = None
) -> tuple[float, float]:
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    best_t, best_score = grid[0], -1.0
    for t in grid:
        preds = _predictions_at(candidates, probs, t)
        score = macro_f05(preds, truth)
        if score > best_score:
            best_t, best_score = t, score
    return best_t, best_score
