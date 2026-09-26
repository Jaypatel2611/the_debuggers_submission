"""Unit tests for src.evaluation.metrics -- PRD §17 macro F0.5 + diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.evaluation.metrics import evaluate, f_beta, macro_f05  # noqa: E402


def test_f_beta_perfect_match_is_one() -> None:
    assert f_beta(1.0, 1.0) == pytest.approx(1.0)


def test_f_beta_zero_precision_and_recall_is_zero() -> None:
    assert f_beta(0.0, 0.0) == 0.0


def test_macro_f05_true_singleton_predicted_empty_scores_one() -> None:
    result = macro_f05({"S1-1": set()}, {"S1-1": set()})
    assert result == pytest.approx(1.0)


def test_macro_f05_true_singleton_predicted_nonempty_scores_zero() -> None:
    result = macro_f05({"S1-1": {"S2-1"}}, {"S1-1": set()})
    assert result == pytest.approx(0.0)


def test_macro_f05_weights_precision_over_recall() -> None:
    # one false positive alongside the true match: precision 0.5, recall 1.0
    result = macro_f05({"S1-1": {"S2-1", "S2-2"}}, {"S1-1": {"S2-1"}})
    assert result == pytest.approx(0.625 / 1.125)


def test_evaluate_reports_singleton_accuracy() -> None:
    preds = {"S1-1": set(), "S1-2": {"S2-1"}}
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    result = evaluate(preds, truth)
    assert result["singleton_accuracy"] == pytest.approx(1.0)
    assert result["macro_f05"] == pytest.approx(1.0)


def test_evaluate_missing_prediction_key_treated_as_empty() -> None:
    # S1-2 never appears in predictions at all -- must be treated as an
    # empty prediction, not crash on a missing dict key.
    result = evaluate({}, {"S1-1": {"S2-1"}})
    assert result["macro_f05"] == pytest.approx(0.0)
