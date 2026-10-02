"""B4 (optional, gated; plan D5b): multilingual sentence embeddings + FAISS top-K per country.
Requires the `dense` extra (sentence-transformers, faiss). GPU is used automatically when available."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..config import DenseBlockerConfig
from ..progress import log, pbar


def dense_blocker(q_texts: list[str], p_texts: list[str], cfg: DenseBlockerConfig, tag: str, out_dir=None):
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("dense blocker needs `uv sync --extra dense`") from e
    model = SentenceTransformer(cfg.model_id)
    prefix = "query: " if "e5" in cfg.model_id else ""

    def encode(texts, what):
        vecs = []
        for i in pbar(range(0, len(texts), cfg.batch_size), desc=f"dense[{tag}] encode {what}", unit="batch"):
            vecs.append(model.encode([prefix + t for t in texts[i:i + cfg.batch_size]], batch_size=cfg.batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
        return np.vstack(vecs).astype(np.float32)

    P = encode(p_texts, "pool"); Q = encode(q_texts, "queries")
    if out_dir is not None:
        np.save(out_dir / f"embeddings_pool_{tag}.npy", P.astype(np.float16))
        np.save(out_dir / f"embeddings_queries_{tag}.npy", Q.astype(np.float16))
    d = P.shape[1]
    index = faiss.IndexFlatIP(d)
    if hasattr(faiss, "StandardGpuResources"):
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception:  # pragma: no cover
            pass
    index.add(P)
    qs, ps, sc, rk = [], [], [], []
    for i in pbar(range(0, Q.shape[0], 4096), desc=f"dense[{tag}] search", unit="batch"):
        D, I = index.search(Q[i:i + 4096], cfg.k)
        for r, (drow, irow) in enumerate(zip(D, I)):
            m = (irow >= 0) & (drow >= cfg.min_cos)
            n = int(m.sum())
            qs.extend([i + r] * n); ps.extend(irow[m].tolist()); sc.extend(drow[m].tolist()); rk.extend(range(1, n + 1))
    log(f"dense[{tag}]: {len(qs):,} hits")
    return pl.DataFrame({"q_idx": np.asarray(qs, np.int64), "p_idx": np.asarray(ps, np.int64),
                         "rank": np.asarray(rk, np.int32), "score": np.asarray(sc, np.float32)})
