"""TSV loaders for the Business Entity Resolution dataset (PRD §2, §21).

Row counts are hard-validated against the values PRD §2 measured directly
against the shipped ``student_resource/dataset/``. A mismatch means a
different dataset, a truncated download, or a mis-parsed file reached the
pipeline -- the problem statement calls a comma-separated file misread as
tab-separated the #1 documented mistake -- so this raises rather than warns.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl

# PRD §2's table reports `wc -l` output, which counts the header line too.
# `pl.read_csv` returns data-row count only (it consumes the header as
# column names), so every value here is the PRD's number minus 1. Caught by
# actually running this loader against the real files -- see CHANGELOG.md.
EXPECTED_ROWS: dict[str, int] = {
    "train_source1": 2_206_821,
    "train_source2": 5_034_616,
    "train_source3": 5_285_603,
    "train_ground_truth": 2_206_821,
    "test_source1": 1_732_544,
    "test_source2": 4_887_273,
    "test_source3": 5_082_316,
}

SOURCE_SCHEMA: dict[str, pl.DataType] = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}

GROUND_TRUTH_SCHEMA: dict[str, pl.DataType] = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
}


class RowCountMismatch(RuntimeError):
    """A loaded file's row count doesn't match PRD §2's measured value."""


class SchemaMismatch(RuntimeError):
    """A loaded file's header columns don't match the expected schema."""


def _read_tsv(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected dataset file not found: {path}")
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,  # addresses/IDs are never CSV-quoted; a literal
        #                   quote char in the data must stay literal.
        schema_overrides=schema,
        empty_string_is_null=False,  # a blank address or an empty
        #                   matched_entity_ids list must read as "", not
        #                   null -- singletons and blank addresses (PRD §9)
        #                   both depend on this.
    )
    expected_cols = list(schema.keys())
    if df.columns != expected_cols:
        raise SchemaMismatch(
            f"{path.name}: expected header {expected_cols}, got {df.columns}."
        )
    return df


def _validate_rows(name: str, df: pl.DataFrame) -> pl.DataFrame:
    expected = EXPECTED_ROWS[name]
    actual = df.height
    if actual != expected:
        raise RowCountMismatch(
            f"{name}: expected {expected:,} rows (PRD §2), got {actual:,}."
        )
    return df


def load_source(path: Path, name: str) -> pl.DataFrame:
    """Load one ``*_source{1,2,3}.tsv`` file and validate its row count."""
    return _validate_rows(name, _read_tsv(path, SOURCE_SCHEMA))


def load_ground_truth(path: Path) -> pl.DataFrame:
    """Load ``train_ground_truth.tsv`` and validate its row count."""
    return _validate_rows("train_ground_truth", _read_tsv(path, GROUND_TRUTH_SCHEMA))


@dataclasses.dataclass(frozen=True)
class BERDataset:
    train_source1: pl.DataFrame
    train_source2: pl.DataFrame
    train_source3: pl.DataFrame
    train_ground_truth: pl.DataFrame
    test_source1: pl.DataFrame
    test_source2: pl.DataFrame
    test_source3: pl.DataFrame


def load_all(data_dir: Path) -> BERDataset:
    """Load and row-count-validate all six source TSVs plus ground truth.

    ``data_dir`` is the challenge's ``dataset/`` folder (contains ``train/``
    and ``test/`` -- i.e. ``student_resource/dataset``).
    """
    train, test = data_dir / "train", data_dir / "test"
    return BERDataset(
        train_source1=load_source(train / "train_source1.tsv", "train_source1"),
        train_source2=load_source(train / "train_source2.tsv", "train_source2"),
        train_source3=load_source(train / "train_source3.tsv", "train_source3"),
        train_ground_truth=load_ground_truth(train / "train_ground_truth.tsv"),
        test_source1=load_source(test / "test_source1.tsv", "test_source1"),
        test_source2=load_source(test / "test_source2.tsv", "test_source2"),
        test_source3=load_source(test / "test_source3.tsv", "test_source3"),
    )


def default_data_dir() -> Path:
    """``student_resource/dataset`` at the repo root, resolved from this file."""
    # this file:  <repo>/code/business_entity_resolution/src/data/loaders.py
    return Path(__file__).resolve().parents[4] / "student_resource" / "dataset"


def _demo() -> None:
    """Self-check: load the real dataset if present and report row counts."""
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data_dir()
    if not data_dir.is_dir():
        print(f"[loaders] dataset dir not found at {data_dir} -- skipping live check")
        return
    ds = load_all(data_dir)
    for field in dataclasses.fields(ds):
        df = getattr(ds, field.name)
        print(f"  {field.name:<20} {df.height:>10,} rows  (expected OK)")
    print("PASS -- all six files + ground truth loaded and row-count-validated")


if __name__ == "__main__":
    _demo()
