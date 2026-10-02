"""GPU top-k cosine retrieval for sparse TF-IDF matrices (torch CSR sparse-matmul + topk).

Benchmarked on an RTX 4060 (20k queries x 412k pool): char 3-gram pass 4.8x faster than the 16-thread CPU
sparse top-k, word pass slower (very sparse rows favour the CPU kernel). Cost per query scales with the pool
size, so it is combined with state sharding. The pool is processed in row blocks that fit in VRAM; queries in
chunks sized by a dense-output budget; per-query top-k is merged across pool blocks. CPU-side chunk preparation
is prefetched on a thread so the GPU never waits for scipy slicing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

from ..progress import log, pbar

_TORCH = None


def torch_module():
    global _TORCH
    if _TORCH is None:
        import torch  # type: ignore
        _TORCH = torch
    return _TORCH


def gpu_available() -> bool:
    try:
        return bool(torch_module().cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def gpu_name() -> str:
    t = torch_module()
    return t.cuda.get_device_name(0) if t.cuda.is_available() else "none"


def _csr_cpu_tensors(m: sp.csr_matrix):
    t = torch_module()
    return (t.from_numpy(m.indptr.astype(np.int64)), t.from_numpy(m.indices.astype(np.int64)),
            t.from_numpy(np.ascontiguousarray(m.data, dtype=np.float32)), m.shape)


def _to_cuda_csr(parts):
    t = torch_module()
    crow, col, val, shape = parts
    return t.sparse_csr_tensor(crow.cuda(non_blocking=True), col.cuda(non_blocking=True), val.cuda(non_blocking=True), size=shape)


def gpu_topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int, min_cos: float, pool_block_rows: int, out_budget_bytes: int,
             query_chunk_max: int, tag: str):
    """Top-k rows of B (pool) for every row of A (queries) by dot product (cosine if rows are l2-normalised).
    Returns (q_idx, p_idx, score) numpy arrays, unsorted within query."""
    t = torch_module()
    nq, npool = A.shape[0], B.shape[0]
    if nq == 0 or npool == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    A = A.tocsr(); B = B.tocsr()
    kk_total = min(k, npool)
    nblocks = (npool + pool_block_rows - 1) // pool_block_rows
    # per query: running best (scores, indices) across pool blocks
    best_s = np.full((nq, kk_total * nblocks), -1.0, dtype=np.float32)
    best_i = np.full((nq, kk_total * nblocks), -1, dtype=np.int64)
    bar = pbar(total=nq * nblocks, desc=f"gpu-topk[{tag}]", unit="q", leave=False)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        for bi, pb in enumerate(range(0, npool, pool_block_rows)):
            Bblk = B[pb:pb + pool_block_rows]
            Bg = _to_cuda_csr(_csr_cpu_tensors(Bblk))
            nb = Bblk.shape[0]
            kb = min(k, nb)
            chunk = int(max(8, min(query_chunk_max, out_budget_bytes // (4 * nb))))
            starts = list(range(0, nq, chunk))
            fut = prefetch.submit(_csr_cpu_tensors, A[starts[0]:starts[0] + chunk])
            for si, qa in enumerate(starts):
                parts = fut.result()
                if si + 1 < len(starts):
                    nxt = starts[si + 1]
                    fut = prefetch.submit(_csr_cpu_tensors, A[nxt:nxt + chunk])
                Ad = _to_cuda_csr(parts).to_dense()                 # (b x V)
                C = t.sparse.mm(Bg, Ad.t())                          # (nb x b)
                Ct = C.t().contiguous()                              # (b x nb)
                v, ix = t.topk(Ct, kb, dim=1)
                v = v.cpu().numpy(); ix = ix.cpu().numpy().astype(np.int64) + pb
                b = v.shape[0]
                col0 = bi * kk_total
                best_s[qa:qa + b, col0:col0 + kb] = v
                best_i[qa:qa + b, col0:col0 + kb] = ix
                del Ad, C, Ct
                bar.update(b)
            del Bg
            t.cuda.empty_cache()
    bar.close()
    # merge across blocks: keep global top-k per query, filter by min_cos
    if nblocks > 1:
        order = np.argsort(-best_s, axis=1)[:, :kk_total]
        best_s = np.take_along_axis(best_s, order, axis=1)
        best_i = np.take_along_axis(best_i, order, axis=1)
    mask = (best_s >= min_cos) & (best_i >= 0)
    q_idx = np.repeat(np.arange(nq, dtype=np.int64), mask.sum(axis=1))
    return q_idx, best_i[mask], best_s[mask]


def gpu_memory_summary() -> str:
    t = torch_module()
    if not t.cuda.is_available():
        return "no cuda"
    return f"peak {t.cuda.max_memory_allocated() / 1e9:.2f} GB"
