"""Global post-processing signals for `export`: uniqueness flags for empty-address candidates.

    uv run python -m business_entity_resolution.ml uniq-flags --artifacts-dir artifacts --split test

Why (valid_llm, 2026-09-27): 75% of the blend's remaining candidate-level errors are pool records with an EMPTY address
and a near-identical name. Pair scorers cannot decide them ("Cinder LLC | <no address>" belongs to S1 "Cinder LLC,
Knoxville" only if no other S1 entity is called Cinder LLC), but the candidate lists can: counted over every S1 entity of
the split, the label rate is 97.6% when exactly one S1 entity has name similarity >= 0.8 to the record and 3.7% when five
or more do. `export --uniqueness` adds the unique ones (p >= 0.3) and drops picks with >= 3 such entities (+0.0016 F0.5
on valid_llm, same rule validated with the name-similarity counts computed here).

Writes <split>/ml/uniq_flags.parquet: (s1_id, cand_id, n_name80) for every candidate pair whose record has an empty
address and name similarity >= 0.8 (max of cos_char_name, cos_word), with n_name80 = number of S1 entities of the split
whose candidate list holds that record at name similarity >= 0.8.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from ..io import load_frame
from ..progress import log
from .common import candidate_part_files, ml_dir

NAME_SIM = 0.8


def run(artifacts_dir: Path, split: str) -> Path:
    t0 = time.time()
    split_dir = artifacts_dir / split
    empties = pl.concat([load_frame(split_dir / f"normalized_source{i}.pkl").select(pl.col("entity_id").alias("cand_id"), pl.col("business_address").fill_null("").str.strip_chars().alias("a"))
                         for i in (2, 3)]).filter(pl.col("a") == "").select("cand_id")
    log(f"[{split}] {empties.height:,} pool records with an empty address")
    parts = candidate_part_files(split_dir)
    ns = pl.max_horizontal(pl.col("cos_char_name").fill_null(0.0), pl.col("cos_word").fill_null(0.0))
    rows = (pl.concat([pl.scan_parquet(f).select("s1_id", "cand_id", "cos_char_name", "cos_word") for f in parts])
              .filter(ns >= NAME_SIM).select("s1_id", "cand_id").collect(engine="streaming"))
    counts = rows.group_by("cand_id").agg(pl.len().cast(pl.Int32).alias("n_name80"))
    flags = rows.join(empties, on="cand_id", how="semi").join(counts, on="cand_id", how="left")
    out = ml_dir(artifacts_dir, split) / "uniq_flags.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    flags.write_parquet(out)
    dist = flags["n_name80"].clip(upper_bound=5).value_counts().sort("n_name80")
    log(f"[{split}] {rows.height:,} candidate pairs at name similarity >= {NAME_SIM}; {flags.height:,} of them with an empty-address record "
        f"-> {out} | S1 entities per record (capped 5): {dict(zip(dist['n_name80'].to_list(), dist['count'].to_list()))} | {time.time() - t0:.0f}s")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--split", default="test", choices=["train", "test"])
    a = p.parse_args(argv)
    run(Path(a.artifacts_dir), a.split)


if __name__ == "__main__":
    main()
