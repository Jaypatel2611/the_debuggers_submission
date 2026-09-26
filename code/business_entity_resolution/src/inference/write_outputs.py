"""matching_results.tsv / candidate_pairs.tsv writer (PRD §22).

Row driver is `all_test_s1_ids`, never a dict's keys, so a silently-dropped
entity can never simply be missing. The subset invariant (matched ids
subset of candidate ids, per entity) is enforced here by intersecting
before writing -- not trusted from the caller.
"""
from __future__ import annotations

from pathlib import Path


def _write_tsv(path: Path, header: str, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for s1_id, joined in rows:
            f.write(f"{s1_id}\t{joined}\n")


def write_outputs(
    all_test_s1_ids: list[str],
    matches: dict[str, set[str]],
    candidate_ids_by_s1: dict[str, set[str]],
    matching_path: Path,
    candidate_path: Path,
) -> None:
    matching_rows = []
    candidate_rows = []
    for s1_id in all_test_s1_ids:
        cands = candidate_ids_by_s1.get(s1_id, set())
        matched = matches.get(s1_id, set()) & cands
        matching_rows.append((s1_id, ",".join(sorted(matched))))
        candidate_rows.append((s1_id, ",".join(sorted(cands))))

    matching_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(matching_path, "source1_entity_id\tmatched_entity_ids", matching_rows)
    _write_tsv(candidate_path, "source1_entity_id\tcandidate_entity_ids", candidate_rows)
