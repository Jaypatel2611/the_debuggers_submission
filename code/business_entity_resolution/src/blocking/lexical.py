"""Exact-match + TF-IDF lexical blocking, partitioned by country (PRD §12
MVP subset for Phase 0 -- no address-component blocking, no bi-encoder, no
cross-country buffer, no fallback blocker: those are Phase 4/5).
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocessing.normalize import normalize_text

try:
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _GPU_AVAILABLE = False

_CANDIDATE_SCHEMA = ["source1_entity_id", "candidate_id", "country", "blocking_rank", "tfidf_score", "exact_match"]


# ponytail: max_features bounds vocab size so fit_transform stays tractable
# at multi-million-row country partitions (US/India); chunk_size bounds the
# peak memory of any single sparse @ sparse matmul instead of doing the
# whole country partition (millions x millions) in one shot. Upgrade path
# if recall suffers: drop max_features, or swap the chunked matmul for a
# proper ANN index (FAISS, already planned for Phase 5's bi-encoder stage).
_MAX_VOCAB_FEATURES = 50_000
_CHUNK_SIZE = 2_000
# ponytail: max_df/min_df only kick in above this candidate-pool size -- a
# tiny pool (unit-test fixtures, a sparse country partition) has no nnz
# blow-up risk and filtering it can empty the vocabulary entirely.
_FILTER_THRESHOLD = 5_000


# ponytail: a bucket can still be much bigger than the 100k-row sample this
# module was benchmarked against (a common prefix like "the"/"inc" at full
# 1.3M-row country scale) -- densifying a whole bucket in one matmul
# measured a real CUDA OOM ("tried to allocate 3.12 GiB, 2.00 GiB free") on
# a T4. Tile both sides instead: bounded (s1_batch x cand_batch) matmuls,
# keep each candidate-batch's local top-k, then merge -- exact top-k, not
# approximate, since the true top-k is always among the union of every
# batch's local top-k.
_GPU_S1_BATCH = 256
_GPU_CAND_BATCH = 4096


def _topk_gpu(s1_matrix: sparse.csr_matrix, cand_matrix_t: sparse.csr_matrix, top_k: int) -> list[list[tuple[int, float]]]:
    """Dense matmul on CUDA, tiled to bound peak GPU memory regardless of
    bucket size."""
    device = torch.device("cuda")
    n_s1, n_cand = s1_matrix.shape[0], cand_matrix_t.shape[1]
    out: list[list[tuple[int, float]]] = []

    for s1_start in range(0, n_s1, _GPU_S1_BATCH):
        s1_chunk = torch.from_numpy(s1_matrix[s1_start:s1_start + _GPU_S1_BATCH].toarray()).to(device)
        batch_vals: list[torch.Tensor] = []
        batch_idx: list[torch.Tensor] = []

        for cand_start in range(0, n_cand, _GPU_CAND_BATCH):
            cand_chunk = torch.from_numpy(
                cand_matrix_t[:, cand_start:cand_start + _GPU_CAND_BATCH].toarray()
            ).to(device)
            sims = s1_chunk @ cand_chunk  # (s1_batch, cand_batch), cosine since both L2-normalized
            k = min(top_k, sims.shape[1])
            if k == 0:
                continue
            vals, idx = torch.topk(sims, k=k, dim=1)
            batch_vals.append(vals)
            batch_idx.append(idx + cand_start)

        if not batch_vals:
            out.extend([] for _ in range(s1_chunk.shape[0]))
            continue

        all_vals = torch.cat(batch_vals, dim=1)
        all_idx = torch.cat(batch_idx, dim=1)
        k_final = min(top_k, all_vals.shape[1])
        final_vals, final_pos = torch.topk(all_vals, k=k_final, dim=1)
        final_idx = torch.gather(all_idx, 1, final_pos)

        vals_np, idx_np = final_vals.cpu().numpy(), final_idx.cpu().numpy()
        out.extend(
            [(int(c), float(v)) for c, v in zip(row_idx, row_vals) if v > 0]
            for row_idx, row_vals in zip(idx_np, vals_np)
        )
    return out


def _tfidf_topk(s1_names: list[str], cand_names: list[str], top_k: int) -> list[list[tuple[int, float]]]:
    """Per S1 row, up to top_k (candidate_row_index, cosine_score) pairs."""
    if len(cand_names) > _FILTER_THRESHOLD:
        # drop near-universal char n-grams (max_df) so the sparse cosine
        # matmul's row density stays bounded at multi-million-row scale.
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=_MAX_VOCAB_FEATURES, max_df=0.3, min_df=2,
        )
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=_MAX_VOCAB_FEATURES)
    cand_matrix = sparse.csr_matrix(vec.fit_transform(cand_names))  # (n_cand, vocab), L2-normalized rows
    s1_matrix = sparse.csr_matrix(vec.transform(s1_names))  # (n_s1, vocab)
    cand_matrix_t = cand_matrix.T.tocsr()

    if _GPU_AVAILABLE:
        return _topk_gpu(s1_matrix, cand_matrix_t, top_k)

    out: list[list[tuple[int, float]]] = []
    for chunk_start in range(0, s1_matrix.shape[0], _CHUNK_SIZE):
        chunk = s1_matrix[chunk_start:chunk_start + _CHUNK_SIZE]
        sims = sparse.csr_matrix(chunk @ cand_matrix_t)  # cosine, since both L2-normalized
        for row_idx in range(sims.shape[0]):
            start, end = sims.indptr[row_idx], sims.indptr[row_idx + 1]
            cols = sims.indices[start:end]
            vals = sims.data[start:end]
            if len(vals) == 0:
                out.append([])
                continue
            order = vals.argsort()[::-1][:top_k]
            out.append([(int(cols[i]), float(vals[i])) for i in order])
    return out


# ponytail: brute-force TF-IDF cosine over a whole country partition
# (measured: 20k x 40k real US rows took ~77s -- full US partition is
# ~1.3M x several-million, which extrapolates to days) is not tractable on
# this hardware. First-2-char prefix bucketing is PRD §12's own listed
# "exact-match ... first-N-chars prefix" method, combined here with TF-IDF
# as its own primary lexical blocker within each bucket -- shrinks the
# per-call matmul from millions x millions down to bucket-sized, at the
# cost of missing a true match whose first two normalized characters
# differ (a typo/abbreviation right at the start of the name). That recall
# gap is exactly what Phase 4 (§19 "blocking recall hardening" -- address
# blocking, cross-country buffer, fallback blocker) exists to close.
def _prefix_key(name_norm: str) -> str:
    return name_norm[:3] if name_norm else ""


def _bucketed_tfidf_topk(
    s1_names_norm: list[str], cand_names_norm: list[str], top_k: int
) -> list[list[tuple[int, float]]]:
    """Same return shape as ``_tfidf_topk`` (per S1 row, list of (candidate
    index into ``cand_names_norm``, cosine_score)), but only ever runs
    TF-IDF within same-prefix-bucket candidate pools."""
    cand_buckets: dict[str, list[int]] = {}
    for j, name in enumerate(cand_names_norm):
        cand_buckets.setdefault(_prefix_key(name), []).append(j)

    s1_buckets: dict[str, list[int]] = {}
    for i, name in enumerate(s1_names_norm):
        s1_buckets.setdefault(_prefix_key(name), []).append(i)

    # ponytail: buckets are independent TF-IDF sub-problems and look
    # parallelizable, but measured attempts made things worse here --
    # process-based joblib (loky) deadlocked under this Windows+Git-Bash
    # setup (worker spawned, near-zero CPU, never returns), and thread-based
    # joblib was slower than plain sequential (40s vs 35s on a 100k x 200k
    # real benchmark) because the per-bucket TfidfVectorizer fit is mostly
    # GIL-held pure-Python tokenization, not matmul. Sequential it is.
    # Upgrade path: batch many buckets into one big padded GPU tensor
    # instead of parallelizing the CPU path further.
    result: list[list[tuple[int, float]]] = [[] for _ in s1_names_norm]
    for prefix, s1_idx_list in s1_buckets.items():
        cand_idx_list = cand_buckets.get(prefix, [])
        if not cand_idx_list:
            continue
        sub_topk = _tfidf_topk(
            [s1_names_norm[i] for i in s1_idx_list],
            [cand_names_norm[j] for j in cand_idx_list],
            top_k,
        )
        for local_i, i in enumerate(s1_idx_list):
            result[i] = [(cand_idx_list[local_j], score) for local_j, score in sub_topk[local_i]]
    return result


def _read_checkpoint(path: Path) -> pl.DataFrame | None:
    """None means "treat as not cached" -- covers a torn/corrupt parquet
    file left behind by a hard crash (power loss) mid-write, not just a
    missing one."""
    try:
        return pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 -- any read failure means recompute, not crash
        print(f"[blocking] checkpoint {path} unreadable ({exc}) -- recomputing", flush=True)
        return None


def _write_checkpoint_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write-then-rename so a checkpoint file only ever exists once fully
    written -- a power loss mid-write leaves the .tmp file, never a
    half-written file at the final name that a later run would try to load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def _empty_candidate_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "source1_entity_id": pl.Utf8, "candidate_id": pl.Utf8, "country": pl.Utf8,
        "blocking_rank": pl.Int64, "tfidf_score": pl.Float64, "exact_match": pl.Boolean,
    })


def generate_candidates(
    s1: pl.DataFrame,
    s2: pl.DataFrame,
    s3: pl.DataFrame,
    top_k: int = 20,
    checkpoint_dir: Path | None = None,
    checkpoint_label: str = "candidates",
) -> pl.DataFrame:
    """Blocking is the multi-hour stage at full dataset scale, and it's
    naturally split into independent per-country partitions -- so when
    ``checkpoint_dir`` is given, each finished country's result is written
    to ``<checkpoint_dir>/<checkpoint_label>_<country>.parquet`` and
    reloaded (skipping recomputation) on a rerun that finds it already
    there. A crash mid-run costs at most the one in-flight country, not
    every country finished before it. No hashing/invalidation of stale
    checkpoints against changed inputs -- delete the directory for a
    genuinely fresh run (Phase 0 MVP scope, not a general cache)."""
    s2 = s2.with_columns(source=pl.lit("S2"))
    s3 = s3.with_columns(source=pl.lit("S3"))
    candidates = pl.concat([s2, s3], how="vertical")

    country_frames: list[pl.DataFrame] = []
    for country in s1["country"].unique().to_list():
        s1_country = s1.filter(pl.col("country") == country)
        cand_country = candidates.filter(pl.col("country") == country)
        if s1_country.height == 0 or cand_country.height == 0:
            continue

        ckpt_path = checkpoint_dir / f"{checkpoint_label}_{country}.parquet" if checkpoint_dir else None
        if ckpt_path is not None and ckpt_path.exists():
            cached = _read_checkpoint(ckpt_path)
            if cached is not None:
                print(f"[blocking] {country}: loaded from checkpoint {ckpt_path}", flush=True)
                country_frames.append(cached)
                continue

        t0 = time.time()
        rows: list[dict] = []
        print(
            f"[blocking] {country}: {s1_country.height} S1 rows x {cand_country.height} candidates...",
            flush=True,
        )

        s1_ids = s1_country["entity_id"].to_list()
        s1_names_norm = [normalize_text(n) for n in s1_country["business_name"].to_list()]
        cand_ids = cand_country["entity_id"].to_list()
        cand_names_norm = [normalize_text(n) for n in cand_country["business_name"].to_list()]

        # hash join for exact-match blocking (name -> candidate_ids) -- O(1)
        # lookup per S1 row, never a full O(n_s1 * n_cand) scan at dataset scale.
        exact_index: dict[str, list[str]] = {}
        for j, cand_name_norm in enumerate(cand_names_norm):
            exact_index.setdefault(cand_name_norm, []).append(cand_ids[j])

        topk = _bucketed_tfidf_topk(s1_names_norm, cand_names_norm, top_k)
        for i, s1_id in enumerate(s1_ids):
            seen: set[str] = set()
            rank = 0
            # exact match first (forced in, regardless of TF-IDF rank)
            for cid in exact_index.get(s1_names_norm[i], []):
                if cid in seen:
                    continue
                rows.append({
                    "source1_entity_id": s1_id, "candidate_id": cid, "country": country,
                    "blocking_rank": rank, "tfidf_score": 1.0, "exact_match": True,
                })
                seen.add(cid)
                rank += 1
            for j, score in topk[i]:
                cand_id = cand_ids[j]
                if cand_id in seen or rank >= top_k:
                    continue
                rows.append({
                    "source1_entity_id": s1_id, "candidate_id": cand_id, "country": country,
                    "blocking_rank": rank, "tfidf_score": score, "exact_match": False,
                })
                seen.add(cand_id)
                rank += 1

        country_df = pl.DataFrame(rows).select(_CANDIDATE_SCHEMA) if rows else _empty_candidate_frame()
        print(
            f"[blocking] {country}: done in {time.time() - t0:.1f}s, {country_df.height} candidate rows",
            flush=True,
        )
        if ckpt_path is not None:
            _write_checkpoint_atomic(country_df, ckpt_path)
        country_frames.append(country_df)

    if not country_frames:
        return _empty_candidate_frame()
    return pl.concat(country_frames)
