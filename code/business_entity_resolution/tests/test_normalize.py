"""Unit tests for src.preprocessing.normalize -- MVP normalization only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.normalize import normalize_text  # noqa: E402


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize_text("  Acme   CORP.  ") == "acme corp"


def test_normalize_strips_punctuation_but_keeps_alnum_and_spaces() -> None:
    assert normalize_text("O'Brien & Sons, Ltd.") == "o brien sons ltd"


def test_normalize_empty_string_stays_empty() -> None:
    assert normalize_text("") == ""


def test_normalize_does_not_lemmatize() -> None:
    # "stores" must stay "stores" -- a lemmatizer would fold it to "store".
    assert normalize_text("Corner Stores") == "corner stores"
