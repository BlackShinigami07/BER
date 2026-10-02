"""TF-IDF vectorisation (fit once per country, transform in a process pool) and CPU sparse top-k retrieval.

The GPU path for char passes lives in gpu_topk.py; `topk` dispatches per blocker kind."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ..progress import log, pbar

_VEC: TfidfVectorizer | None = None


def make_vectorizer(kind: str, max_df: float = 1.0) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32, min_df=1, max_df=max_df)
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, min_df=2)
    raise ValueError(kind)


def fit_vectorizer(kind: str, texts: list[str], max_df: float, fit_sample: int, seed: int, tag: str) -> TfidfVectorizer:
    t0 = time.time()
    vec = make_vectorizer(kind, max_df)
    if kind == "char" and len(texts) > fit_sample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(texts), size=fit_sample, replace=False)
        vec.fit([texts[i] for i in sel])
        how = f"fit on {fit_sample:,} sampled rows"
    else:
        vec.fit(texts)
        how = f"fit on all {len(texts):,} rows"
    log(f"tfidf[{tag}] {kind}: vocab {len(vec.vocabulary_):,} ({how}) in {time.time() - t0:.0f}s")
    return vec


def _init_transform(vec):
    global _VEC
    _VEC = vec


def _transform_chunk(texts: list[str]) -> sp.csr_matrix:
    return _VEC.transform(texts).tocsr()


def parallel_transform(vec: TfidfVectorizer, texts: list[str], workers: int, chunk: int, tag: str) -> sp.csr_matrix:
    """Transform `texts` with a fitted vectorizer using a process pool (sklearn analyzers are single-threaded Python)."""
    if not texts:
        return sp.csr_matrix((0, len(vec.vocabulary_)), dtype=np.float32)
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    if workers <= 1 or len(chunks) == 1:
        _init_transform(vec)
        mats = [_transform_chunk(c) for c in pbar(chunks, desc=f"transform[{tag}]", unit="chunk", leave=False)]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks)), initializer=_init_transform, initargs=(vec,)) as ex:
            mats = list(pbar(ex.map(_transform_chunk, chunks), total=len(chunks), desc=f"transform[{tag}]", unit="chunk", leave=False))
    M = sp.vstack(mats, format="csr")
    M.indices = M.indices.astype(np.int32, copy=False); M.indptr = M.indptr.astype(np.int64 if M.nnz > 2**31 - 1 else np.int32, copy=False)
    M.data = M.data.astype(np.float32, copy=False)
    return M


def cpu_topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int, min_cos: float, chunk_size: int, threads: int, tag: str):
    """sparse_dot_topn top-k of rows of A against rows of B. Returns (q_idx, p_idx, score) arrays."""
    nq = A.shape[0]
    if nq == 0 or B.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    A = _int32(A.tocsr()); BT = _int32(B.T.tocsr())
    qs, ps, sc = [], [], []
    bar = pbar(total=nq, desc=f"cpu-topk[{tag}]", unit="q", leave=False)
    for start in range(0, nq, chunk_size):
        Ac = A[start:start + chunk_size]
        C = sp_matmul_topn(Ac, BT, top_n=k, threshold=min_cos, sort=False, n_threads=threads).tocsr()
        rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
        qs.append(rows + start); ps.append(C.indices.astype(np.int64)); sc.append(C.data.astype(np.float32))
        bar.update(Ac.shape[0])
    bar.close()
    return (np.concatenate(qs) if qs else np.zeros(0, np.int64), np.concatenate(ps) if ps else np.zeros(0, np.int64),
            np.concatenate(sc) if sc else np.zeros(0, np.float32))


def _int32(m: sp.csr_matrix) -> sp.csr_matrix:
    if m.indices.dtype != np.int32 or m.indptr.dtype != np.int32:
        m = m.copy(); m.indices = m.indices.astype(np.int32); m.indptr = m.indptr.astype(np.int32)
    if m.data.dtype != np.float32:
        m.data = m.data.astype(np.float32)
    return m


def hits_frame(q_idx: np.ndarray, p_idx: np.ndarray, score: np.ndarray, k: int, q_map: np.ndarray | None, p_map: np.ndarray | None) -> pl.DataFrame:
    """Rank hits per query (1 = best), keep <= k, map local to global indices."""
    if q_map is not None and len(q_idx):
        q_idx = q_map[q_idx]
    if p_map is not None and len(p_idx):
        p_idx = p_map[p_idx]
    df = pl.DataFrame({"q_idx": q_idx.astype(np.int64), "p_idx": p_idx.astype(np.int64), "score": score.astype(np.float32)})
    if df.height == 0:
        return df.with_columns(pl.Series("rank", [], dtype=pl.Int32))
    df = df.unique(subset=["q_idx", "p_idx"], keep="first")
    df = df.with_columns(pl.col("score").rank(method="ordinal", descending=True).over("q_idx").cast(pl.Int32).alias("rank"))
    return df.filter(pl.col("rank") <= k)
