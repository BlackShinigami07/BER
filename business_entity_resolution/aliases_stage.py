"""Stage `aliases` (train only): learn state/city alias tables from ground-truth pairs -> artifacts/train/aliases.pkl."""
from __future__ import annotations

import polars as pl

from .aliases import learn_all_state_aliases
from .config import BlockingConfig
from .io import gt_to_pairs, list_countries, normalized_part, read_ground_truth, save_pkl
from .progress import log


def stage_aliases(cfg: BlockingConfig) -> None:
    d = cfg.split_dir("train")
    countries = list_countries(d)
    cols = ["entity_id", "country", "addr_comps"]
    s1 = pl.concat([pl.read_parquet(normalized_part(d, "source1", c), columns=cols) for c in countries])
    pool = pl.concat([pl.read_parquet(normalized_part(d, s, c), columns=cols) for c in countries for s in ("source2", "source3")])
    pairs = gt_to_pairs(read_ground_truth(cfg.data_dir / "train" / "train_ground_truth.tsv"))
    if pairs.height > cfg.alias_max_pairs:
        pairs = pairs.sample(n=cfg.alias_max_pairs, seed=cfg.seed)
    log(f"aliases: learning from {pairs.height:,} GT pairs")
    tables = learn_all_state_aliases(s1, pool, pairs)
    save_pkl(tables, d / "aliases.pkl")
