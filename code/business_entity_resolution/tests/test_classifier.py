"""Unit tests for src.models.classifier -- LightGBM pairwise classifier."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.features.pairwise import FEATURE_COLUMNS  # noqa: E402
from src.models.classifier import predict_proba, train_classifier  # noqa: E402


def _labeled_df(n_pos: int = 200, n_neg: int = 200) -> pl.DataFrame:
    random.seed(0)
    rows = []
    for _ in range(n_pos):
        rows.append({**{c: random.uniform(0.8, 1.0) for c in FEATURE_COLUMNS}, "label": 1})
    for _ in range(n_neg):
        rows.append({**{c: random.uniform(0.0, 0.2) for c in FEATURE_COLUMNS}, "label": 0})
    return pl.DataFrame(rows)


def test_train_and_predict_separates_classes() -> None:
    df = _labeled_df()
    model = train_classifier(df, seed=42)
    probs = predict_proba(model, df)
    pos_mean = probs[df["label"].to_numpy() == 1].mean()
    neg_mean = probs[df["label"].to_numpy() == 0].mean()
    assert pos_mean > neg_mean
