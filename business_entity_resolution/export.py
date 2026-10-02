"""Stage `export` (test): write output/candidate_pairs.tsv from the candidate parts (streaming, one part at a time)
plus an empty matching_results stub so the official validator can run."""
from __future__ import annotations

import polars as pl

from .config import BlockingConfig
from .io import candidate_parts, read_source, source_path
from .progress import log, pbar


def stage_export(cfg: BlockingConfig, split: str = "test") -> None:
    d = cfg.split_dir(split)
    s1_ids = read_source(source_path(cfg.data_dir, split, "source1"))["entity_id"].to_list()
    out = cfg.output_dir; out.mkdir(parents=True, exist_ok=True)
    path = out / "candidate_pairs.tsv"
    written: set[str] = set(); n_ids = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for part in pbar(candidate_parts(d), desc="write candidate_pairs.tsv", unit="part"):
            df = (pl.read_parquet(part, columns=["s1_id", "cand_id", "rrf_rank"]).sort(["s1_id", "rrf_rank"])
                  .group_by("s1_id", maintain_order=True).agg(pl.col("cand_id").unique(maintain_order=True)))
            for s, ids in zip(df["s1_id"].to_list(), df["cand_id"].to_list()):
                if s in written:
                    continue  # an S1 entity belongs to exactly one shard; guard anyway
                ids = [x for x in ids if x.startswith(("S2-", "S3-"))]
                f.write(f"{s}\t{','.join(ids)}\n"); written.add(s); n_ids += len(ids)
        missing = [s for s in s1_ids if s not in written]
        for s in missing:
            f.write(f"{s}\t\n")
    log(f"wrote {path}: {len(s1_ids):,} rows ({len(missing):,} without candidates), {n_ids:,} candidate ids")
    stub = out / "matching_results_stub.tsv"
    with open(stub, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1_ids:
            f.write(f"{s}\t\n")
    log(f"wrote {stub} (EMPTY placeholder for the validator; the real matching_results.tsv comes from the ML stage)")
