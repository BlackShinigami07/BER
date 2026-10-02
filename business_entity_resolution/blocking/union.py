"""Union of blocker hits with weighted Reciprocal Rank Fusion and a per-S1 cap (plan D4).

Implemented as full outer joins on (q_idx, p_idx) across blockers (hash joins scale to 1e8 rows), then
fused = sum_b w_b / (rrf_k + rank_b) over the blockers that retrieved the pair, ranked per query, capped."""
from __future__ import annotations

import polars as pl

from ..config import BLOCKER_NAMES


def rrf_union(hits: dict[str, pl.DataFrame], weights: dict[str, float], rrf_k: int, cap: int) -> pl.DataFrame:
    """hits: blocker name -> DataFrame[q_idx, p_idx, rank, score]. Returns one row per (q_idx, p_idx) with
    provenance columns rank_<b>/cos_<b> for every blocker name, n_blockers, rrf_score, rrf_rank (<= cap)."""
    wide: pl.DataFrame | None = None
    present = []
    for b in BLOCKER_NAMES:
        df = hits.get(b)
        if df is None or df.height == 0:
            continue
        present.append(b)
        part = df.select(
            pl.col("q_idx").cast(pl.Int64), pl.col("p_idx").cast(pl.Int64),
            pl.col("rank").cast(pl.Int32).alias(f"rank_{b}"), pl.col("score").cast(pl.Float32).alias(f"cos_{b}"),
        ).unique(subset=["q_idx", "p_idx"], keep="first")
        wide = part if wide is None else wide.join(part, on=["q_idx", "p_idx"], how="full", coalesce=True)
    if wide is None:
        return _empty()
    for b in BLOCKER_NAMES:
        if b not in present:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Int32).alias(f"rank_{b}"), pl.lit(None, dtype=pl.Float32).alias(f"cos_{b}"))
    fused = pl.lit(0.0, dtype=pl.Float32)
    n_bl = pl.lit(0, dtype=pl.Int8)
    for b in BLOCKER_NAMES:
        w = float(weights.get(b, 0.0))
        r = pl.col(f"rank_{b}")
        fused = fused + pl.when(r.is_null()).then(0.0).otherwise(w / (rrf_k + r.cast(pl.Float32)))
        n_bl = n_bl + r.is_not_null().cast(pl.Int8)
    wide = wide.with_columns(fused.cast(pl.Float32).alias("rrf_score"), n_bl.alias("n_blockers"))
    wide = wide.with_columns(
        pl.col("rrf_score").rank(method="ordinal", descending=True).over("q_idx").cast(pl.Int32).alias("rrf_rank"))
    return wide.filter(pl.col("rrf_rank") <= cap).sort(["q_idx", "rrf_rank"])


def _empty() -> pl.DataFrame:
    cols = {"q_idx": pl.Int64, "p_idx": pl.Int64}
    for b in BLOCKER_NAMES:
        cols[f"rank_{b}"] = pl.Int32; cols[f"cos_{b}"] = pl.Float32
    cols.update({"rrf_score": pl.Float32, "n_blockers": pl.Int8, "rrf_rank": pl.Int32})
    return pl.DataFrame({k: pl.Series([], dtype=v) for k, v in cols.items()})
