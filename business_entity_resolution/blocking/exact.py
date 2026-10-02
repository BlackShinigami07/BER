"""B0: exact match on the suffix-stripped core name (dict lookup). Blocks bigger than `block_cap` are skipped:
those are generic names ("primary care group") that only the address-aware blockers can rank."""
from __future__ import annotations

import collections

import numpy as np
import polars as pl

from ..progress import log, pbar


def exact_blocker(q_keys: list[str], p_keys: list[str], block_cap: int) -> pl.DataFrame:
    index: dict[str, list[int]] = collections.defaultdict(list)
    for i, k in enumerate(pbar(p_keys, desc="exact: index pool", unit="row", leave=False)):
        if k:
            index[k].append(i)
    qs, ps, rk = [], [], []
    skipped = 0
    for qi, k in enumerate(pbar(q_keys, desc="exact: lookup", unit="row", leave=False)):
        block = index.get(k)
        if not block:
            continue
        if len(block) > block_cap:
            skipped += 1
            continue
        qs.extend([qi] * len(block)); ps.extend(block); rk.extend(range(1, len(block) + 1))
    if skipped:
        log(f"exact: {skipped:,} queries skipped because their name block exceeds {block_cap} pool rows")
    return pl.DataFrame({
        "q_idx": np.asarray(qs, dtype=np.int64), "p_idx": np.asarray(ps, dtype=np.int64),
        "rank": np.asarray(rk, dtype=np.int32), "score": np.ones(len(qs), dtype=np.float32),
    })
