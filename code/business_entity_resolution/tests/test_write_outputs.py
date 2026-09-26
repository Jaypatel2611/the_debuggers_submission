"""Unit tests for src.inference.write_outputs -- PRD §22 writer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.write_outputs import write_outputs  # noqa: E402


def test_every_test_s1_id_gets_exactly_one_row(tmp_path: Path) -> None:
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1", "S1-2", "S1-3"],
        matches={"S1-1": {"S2-1"}},
        candidate_ids_by_s1={"S1-1": {"S2-1", "S2-2"}, "S1-2": set()},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    lines = matching_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 4  # header + 3 entities
    body_ids = {ln.split("\t")[0] for ln in lines[1:]}
    assert body_ids == {"S1-1", "S1-2", "S1-3"}


def test_empty_match_serializes_as_nothing_after_tab(tmp_path: Path) -> None:
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1"], matches={}, candidate_ids_by_s1={},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    line = matching_path.read_text(encoding="utf-8").splitlines()[1]
    assert line == "S1-1\t"


def test_match_not_in_candidates_is_dropped_not_written(tmp_path: Path) -> None:
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    # S2-99 is claimed as a match but was never a candidate -- must not appear.
    write_outputs(
        all_test_s1_ids=["S1-1"], matches={"S1-1": {"S2-99"}},
        candidate_ids_by_s1={"S1-1": {"S2-1"}},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    line = matching_path.read_text(encoding="utf-8").splitlines()[1]
    assert line == "S1-1\t"


def test_subset_invariant_holds_across_full_file(tmp_path: Path) -> None:
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1"], matches={"S1-1": {"S2-1"}},
        candidate_ids_by_s1={"S1-1": {"S2-1", "S2-2"}},
        matching_path=matching_path, candidate_path=candidate_path,
    )

    def _parse(path: Path) -> dict[str, set[str]]:
        out = {}
        for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = ln.split("\t")
            s1_id = parts[0]
            ids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            out[s1_id] = set(ids)
        return out

    m, c = _parse(matching_path), _parse(candidate_path)
    for s1_id, matched in m.items():
        assert matched <= c.get(s1_id, set())


def test_orphan_country_s1_entity_still_gets_empty_row(tmp_path: Path) -> None:
    # An S1 entity whose country has zero candidates at all -- must still
    # get a row (empty match list), never be silently dropped.
    matching_path = tmp_path / "matching_results.tsv"
    candidate_path = tmp_path / "candidate_pairs.tsv"
    write_outputs(
        all_test_s1_ids=["S1-1", "S1-orphan"],
        matches={"S1-1": {"S2-1"}},
        candidate_ids_by_s1={"S1-1": {"S2-1"}},
        matching_path=matching_path, candidate_path=candidate_path,
    )
    lines = {ln.split("\t")[0]: ln for ln in matching_path.read_text(encoding="utf-8").splitlines()[1:]}
    assert lines["S1-orphan"] == "S1-orphan\t"
