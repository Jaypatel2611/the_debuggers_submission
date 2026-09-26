"""Unit tests for src.data.loaders -- schema/row-count validation logic.

Uses small synthetic TSVs (not the real ~2-5M row dataset) so these run in
milliseconds; the real dataset is exercised separately via
`python -m src.data.loaders` (PRD §19 Phase 0 exit criterion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/business_entity_resolution

from src.data.loaders import (  # noqa: E402
    RowCountMismatch,
    SchemaMismatch,
    load_ground_truth,
    load_source,
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_source_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.loaders as loaders

    # Real EXPECTED_ROWS["train_source1"] is 2_206_822; this fixture only has
    # 1 row on purpose, so point the expectation at the fixture's own size --
    # the mismatch case is covered separately in test_row_count_mismatch_raises.
    monkeypatch.setitem(loaders.EXPECTED_ROWS, "train_source1", 1)
    path = _write(
        tmp_path,
        "train_source1.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Corp\t1 Main St, Springfield\tUS\n",
    )
    df = load_source(path, "train_source1")
    assert df.height == 1
    assert df["business_name"][0] == "Acme Corp"


def test_row_count_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.loaders as loaders

    monkeypatch.setitem(loaders.EXPECTED_ROWS, "train_source1", 999)
    path = _write(
        tmp_path,
        "train_source1.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Corp\t1 Main St\tUS\n",
    )
    with pytest.raises(RowCountMismatch):
        load_source(path, "train_source1")


def test_schema_mismatch_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "train_source1.tsv",
        "id\tname\taddress\tcountry\n"
        "S1-1\tAcme Corp\t1 Main St\tUS\n",
    )
    with pytest.raises(SchemaMismatch):
        load_source(path, "train_source1")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_source(tmp_path / "does_not_exist.tsv", "train_source1")


def test_blank_address_reads_as_empty_string_not_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.loaders as loaders

    monkeypatch.setitem(loaders.EXPECTED_ROWS, "train_source1", 1)
    path = _write(
        tmp_path,
        "train_source1.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Corp\t\tUS\n",
    )
    df = load_source(path, "train_source1")
    assert df["business_address"][0] == ""
    assert df["business_address"][0] is not None


def test_ground_truth_empty_matched_ids_is_empty_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.loaders as loaders

    monkeypatch.setitem(loaders.EXPECTED_ROWS, "train_ground_truth", 2)
    path = _write(
        tmp_path,
        "train_ground_truth.tsv",
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n",
    )
    df = load_ground_truth(path)
    assert df["matched_entity_ids"][1] == ""
