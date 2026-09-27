"""PRD §22 single entry point: raw TSV -> validated output/.

python -m src.run_pipeline [data_dir] [output_dir]
"""
from __future__ import annotations

import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from .blocking.lexical import _read_checkpoint, _write_checkpoint_atomic, generate_candidates
from .data.loaders import BERDataset, default_data_dir, load_all
from .evaluation.metrics import evaluate
from .features.pairwise import build_features
from .inference.assignment import resolve_1to1
from .inference.threshold import sweep_threshold
from .inference.write_outputs import write_outputs
from .models.classifier import predict_proba, train_classifier

SEED = 42


def _label_training_pairs(candidates: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    gt = ground_truth.select(
        "source1_entity_id",
        pl.col("matched_entity_ids").str.split(",").alias("mid"),
    ).explode("mid", empty_as_null=False)
    positive_pairs = set(zip(gt["source1_entity_id"].to_list(), gt["mid"].to_list()))
    is_positive = [
        1 if (s1, cid) in positive_pairs else 0
        for s1, cid in zip(candidates["source1_entity_id"].to_list(), candidates["candidate_id"].to_list())
    ]
    return candidates.with_columns(label=pl.Series(is_positive))


def _stage_start(label: str) -> tuple[str, float]:
    print(f"[pipeline] {label}...", flush=True)
    return label, time.time()


def _stage_done(stage: tuple[str, float]) -> None:
    label, t0 = stage
    print(f"[pipeline] {label} done in {time.time() - t0:.1f}s", flush=True)


def _load_or_build(ckpt_path: Path, label: str, build) -> pl.DataFrame:
    """Whole-stage checkpoint -- coarser than blocking's per-country resume,
    but feature-building is a single call over the whole candidate frame
    (not naturally per-country), and cheap enough relative to blocking that
    one retry on a crash is tolerable. Skipped entirely when ckpt_path is
    None (no checkpoint_dir given)."""
    if ckpt_path is not None and ckpt_path.exists():
        cached = _read_checkpoint(ckpt_path)
        if cached is not None:
            print(f"[pipeline] {label}: loaded from checkpoint {ckpt_path}", flush=True)
            return cached
    stage = _stage_start(label)
    result = build()
    _stage_done(stage)
    if ckpt_path is not None:
        _write_checkpoint_atomic(result, ckpt_path)
    return result


def run(data_dir: Path, output_dir: Path, repo_root: Path, checkpoint_dir: Path | None = None) -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    if checkpoint_dir is not None:
        print(f"[pipeline] checkpointing to {checkpoint_dir} -- rerun with the same output_dir to resume", flush=True)

    stage = _stage_start("loading dataset")
    ds: BERDataset = load_all(data_dir)
    _stage_done(stage)

    train_candidates = generate_candidates(
        ds.train_source1, ds.train_source2, ds.train_source3, top_k=20,
        checkpoint_dir=checkpoint_dir, checkpoint_label="train_blocking",
    )
    train_candidates = _label_training_pairs(train_candidates, ds.train_ground_truth)
    print(f"[pipeline] train candidate rows: {train_candidates.height}", flush=True)

    train_features = _load_or_build(
        checkpoint_dir / "train_features.parquet" if checkpoint_dir else None,
        "building training features",
        lambda: build_features(train_candidates, ds.train_source1, ds.train_source2, ds.train_source3),
    )

    print("[pipeline] naive random row split (PRD §19 Phase 0 -- GroupKFold is Phase 2)...", flush=True)
    train_df, val_df = train_test_split(
        train_features, test_size=0.2, random_state=SEED, stratify=train_features["label"].to_numpy()
    )

    print("[pipeline] training classifier...", flush=True)
    model = train_classifier(train_df, seed=SEED)

    val_probs = predict_proba(model, val_df)
    val_s1_ids = set(val_df["source1_entity_id"].to_list())
    val_truth = {
        s1: (set(m.split(",")) if m else set())
        for s1, m in zip(
            ds.train_ground_truth["source1_entity_id"].to_list(),
            ds.train_ground_truth["matched_entity_ids"].to_list(),
        )
        if s1 in val_s1_ids
    }
    best_threshold, best_val_f05 = sweep_threshold(val_df, val_probs, val_truth)
    print(f"[pipeline] naive-split local macro F0.5 = {best_val_f05:.4f} at threshold {best_threshold}", flush=True)
    metrics = evaluate(resolve_1to1(val_df, val_probs, best_threshold), val_truth)
    print(f"[pipeline] full val metrics: {metrics}", flush=True)

    test_candidates = generate_candidates(
        ds.test_source1, ds.test_source2, ds.test_source3, top_k=20,
        checkpoint_dir=checkpoint_dir, checkpoint_label="test_blocking",
    )
    print(f"[pipeline] test candidate rows: {test_candidates.height}", flush=True)

    test_features = _load_or_build(
        checkpoint_dir / "test_features.parquet" if checkpoint_dir else None,
        "building test features",
        lambda: build_features(test_candidates, ds.test_source1, ds.test_source2, ds.test_source3),
    )

    test_probs = predict_proba(model, test_features)
    test_matches = resolve_1to1(test_features, test_probs, best_threshold)

    candidate_ids_by_s1: dict[str, set[str]] = {}
    for s1_id, cand_id in zip(test_candidates["source1_entity_id"].to_list(), test_candidates["candidate_id"].to_list()):
        candidate_ids_by_s1.setdefault(s1_id, set()).add(cand_id)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    write_outputs(ds.test_source1["entity_id"].to_list(), test_matches, candidate_ids_by_s1, matching_path, candidate_path)
    print(f"[pipeline] wrote {matching_path} and {candidate_path}", flush=True)

    print("[pipeline] running validate_submission.py...", flush=True)
    result = subprocess.run(
        [
            sys.executable, str(repo_root / "student_resource" / "utils" / "validate_submission.py"),
            "--matching", str(matching_path), "--candidate", str(candidate_path),
            "--test-dir", str(data_dir / "test"),
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("[pipeline] validate_submission.py FAILED -- submission not safe to submit")
        return result.returncode
    print("[pipeline] PASS -- validate_submission.py green")
    return 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data_dir()
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else repo_root / "output"
    checkpoint_dir = output_dir / "checkpoints"
    sys.exit(run(data_dir, output_dir, repo_root, checkpoint_dir))


if __name__ == "__main__":
    main()
