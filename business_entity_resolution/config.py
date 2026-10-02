"""All tunable knobs of the blocking pipeline in one place."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TfidfBlockerConfig:
    enabled: bool = True
    k: int = 50            # top-K candidates per S1 entity
    min_cos: float = 0.05  # discard hits below this cosine


@dataclass
class DenseBlockerConfig:
    enabled: bool = False  # gated (see BLOCKING_PLAN.md D5b)
    k: int = 20
    min_cos: float = 0.6
    model_id: str = "intfloat/multilingual-e5-small"
    batch_size: int = 512


@dataclass
class BlockingConfig:
    # paths
    data_dir: Path = Path("dataset")
    artifacts_dir: Path = Path("artifacts")
    output_dir: Path = Path("output")

    # reproducibility / splits
    seed: int = 42
    valid_fraction: float = 0.10      # ml_valid share of S1 entities (stratified by country)

    # sub-sampling for development runs (None = full data)
    sample_s1: int | None = None      # number of S1 entities (stratified by country)
    pool_fraction: float = 1.0        # fraction of the non-matching S2/S3 pool kept when sampling

    # parallelism
    threads: int = 0                  # 0 = all logical CPUs (CPU sparse top-k)
    workers: int = 0                  # 0 = min(8, CPUs) processes for normalisation / TF-IDF transforms
    chunk_size: int = 50_000          # S1 rows per CPU sparse top-k chunk
    norm_chunk_size: int = 200_000    # rows per normalisation chunk
    transform_chunk: int = 200_000    # rows per TF-IDF transform chunk (process pool)

    # GPU (torch + CUDA) for the char 3-gram passes; None = auto-detect
    gpu: bool | None = None
    gpu_pool_block_rows: int = 3_000_000   # pool rows resident on the GPU at once
    gpu_out_budget_bytes: int = 1_000_000_000  # size of the dense score block per query chunk
    gpu_query_chunk_max: int = 256

    # blockers
    exact_enabled: bool = True
    exact_block_cap: int = 300        # skip exact-name blocks larger than this (generic names)
    word: TfidfBlockerConfig = field(default_factory=lambda: TfidfBlockerConfig(True, 50, 0.05))
    char_name: TfidfBlockerConfig = field(default_factory=lambda: TfidfBlockerConfig(True, 30, 0.30))
    char_addr: TfidfBlockerConfig = field(default_factory=lambda: TfidfBlockerConfig(True, 30, 0.30))
    dense: DenseBlockerConfig = field(default_factory=DenseBlockerConfig)
    shard_by_state: bool = True       # state shards for countries with a learned alias table
    word_max_df: float = 0.05         # drop tokens present in > 5% of a country pool (near-zero IDF, huge posting lists)
    char_fit_sample: int = 1_000_000  # rows used to fit the char 3-gram vectorizer (IDF); transform uses all rows

    # union
    rrf_k: int = 60
    rrf_weights: dict[str, float] = field(default_factory=lambda: {
        "exact": 3.0, "word": 2.0, "char_name": 1.5, "char_addr": 1.5, "dense": 0.5})
    cap: int = 60                     # max candidates per S1 entity after RRF
    report_caps: tuple[int, ...] = (10, 20, 30, 50, 60)

    # outputs
    pkl_part_rows: int = 25_000_000   # rows per candidates pkl part (keeps the pandas conversion small)
    keep_hits: bool = False           # keep per-blocker hit files after the union (debugging)
    alias_max_pairs: int = 3_000_000  # GT pairs used to learn alias tables (memory bound)
    export_parquet: bool = True       # parquet parts are always written; this keeps them after the pkl parts exist

    # ---- derived helpers ----
    def split_dir(self, split: str) -> Path:
        return self.artifacts_dir / split

    def n_threads(self) -> int:
        return self.threads or (os.cpu_count() or 4)

    def n_workers(self) -> int:
        return self.workers or max(1, min(8, os.cpu_count() or 1))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d


BLOCKER_NAMES = ("exact", "word", "char_name", "char_addr", "dense")
