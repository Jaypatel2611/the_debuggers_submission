"""LightGBM pairwise-match classifier -- PRD §14 MVP: one model, no
ensembling, no hard-negative mining loop (Phase 8), no singleton
pre-filter (Phase 8). Training is cheap (single boosted-tree fit on a
compact feature matrix) so this stays CPU-only -- no GPU build needed,
unlike blocking's TF-IDF matmul.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from ..features.pairwise import FEATURE_COLUMNS

SEED = 42


def train_classifier(train_df: pl.DataFrame, label_col: str = "label", seed: int = SEED) -> lgb.Booster:
    X = train_df.select(FEATURE_COLUMNS).to_numpy()
    y = train_df[label_col].to_numpy()
    dataset = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
    params = {
        "objective": "binary",
        "is_unbalance": True,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
    }
    return lgb.train(params, dataset, num_boost_round=100)


def predict_proba(model: lgb.Booster, df: pl.DataFrame) -> np.ndarray:
    X = df.select(FEATURE_COLUMNS).to_numpy()
    return model.predict(X)
