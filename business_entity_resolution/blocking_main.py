"""Single runner for the blocking pipeline.

    uv run python -m business_entity_resolution.blocking_main --split train|test|both [--sample N] [--pool-fraction F]
        [--stages prepare,aliases,block,evaluate,export] [--resume] [--threads N] [--workers N] [--cap N]
        [--gpu | --no-gpu] [--dense] [--keep-hits]

train: prepare -> aliases -> block -> evaluate      test: prepare -> block -> export
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import BlockingConfig
from .progress import fmt_secs, log, stage

TRAIN_STAGES = ["prepare", "aliases", "block", "evaluate"]
TEST_STAGES = ["prepare", "block", "export"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blocking / candidate generation pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--split", choices=["train", "test", "both"], required=True)
    p.add_argument("--stages", default=None, help="comma-separated subset of stages to run (default: all for the split)")
    p.add_argument("--resume", action="store_true", help="skip stages whose artifacts already exist")
    p.add_argument("--sample", type=int, default=None, help="development mode: number of S1 entities (stratified by country)")
    p.add_argument("--pool-fraction", type=float, default=None, help="fraction of the non-matching S2/S3 pool to keep")
    p.add_argument("--threads", type=int, default=None, help="threads for the CPU sparse top-k (default: all CPUs)")
    p.add_argument("--workers", type=int, default=None, help="processes for normalisation / TF-IDF transforms (default: min(8, CPUs))")
    p.add_argument("--cap", type=int, default=None, help="max candidates per S1 after RRF")
    p.add_argument("--k-word", type=int, default=None); p.add_argument("--k-char-name", type=int, default=None); p.add_argument("--k-char-addr", type=int, default=None)
    p.add_argument("--no-shard", action="store_true", help="disable state sharding")
    p.add_argument("--word-max-df", type=float, default=None, help="drop word tokens present in more than this fraction of a country pool (1.0 = no pruning)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="require the GPU for char passes (default: auto-detect)")
    g.add_argument("--no-gpu", action="store_true", help="force CPU for everything")
    p.add_argument("--dense", action="store_true", help="enable the optional dense (embedding) blocker B4")
    p.add_argument("--keep-hits", action="store_true", help="keep per-blocker hit files (debugging)")
    p.add_argument("--data-dir", default="dataset"); p.add_argument("--artifacts-dir", default="artifacts"); p.add_argument("--output-dir", default="output")
    p.add_argument("--log-file", default=None, help="also write all progress/log output to this file (works in PowerShell without 2>&1)")
    return p


class _Tee:
    """Duplicate everything written to stderr into a log file (progress bars included)."""

    def __init__(self, stream, path):
        self.stream = stream; self.f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s); self.f.write(s)

    def flush(self):
        self.stream.flush(); self.f.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def config_from_args(a: argparse.Namespace) -> BlockingConfig:
    cfg = BlockingConfig(data_dir=Path(a.data_dir), artifacts_dir=Path(a.artifacts_dir), output_dir=Path(a.output_dir))
    if a.sample is not None: cfg.sample_s1 = a.sample
    if a.pool_fraction is not None: cfg.pool_fraction = a.pool_fraction
    if a.threads is not None: cfg.threads = a.threads
    if a.workers is not None: cfg.workers = a.workers
    if a.cap is not None: cfg.cap = a.cap
    if a.k_word is not None: cfg.word.k = a.k_word
    if a.k_char_name is not None: cfg.char_name.k = a.k_char_name
    if a.k_char_addr is not None: cfg.char_addr.k = a.k_char_addr
    if a.no_shard: cfg.shard_by_state = False
    if a.word_max_df is not None: cfg.word_max_df = a.word_max_df
    if a.gpu: cfg.gpu = True
    if a.no_gpu: cfg.gpu = False
    if a.dense: cfg.dense.enabled = True
    if a.keep_hits: cfg.keep_hits = True
    return cfg


def artifacts_exist(cfg: BlockingConfig, split: str, st: str) -> bool:
    d = cfg.split_dir(split)
    need = {
        "prepare": [d / "countries.json"] + [d / f"normalized_{s}.pkl" for s in ("source1", "source2", "source3")],
        "aliases": [cfg.split_dir("train") / "aliases.pkl"],
        "block": [d / "candidates_index.json", d / "block_summary.json"],
        "evaluate": [d / "blocking_metrics.json"],
        "export": [cfg.output_dir / "candidate_pairs.tsv"],
    }[st]
    return all(p.exists() for p in need)


def run_split(cfg: BlockingConfig, split: str, stages: list[str], resume: bool, summary: list[str]) -> None:
    from .blocking.pipeline import stage_block
    from .prepare import stage_prepare
    for st in stages:
        if resume and artifacts_exist(cfg, split, st):
            log(f"[{split}] stage {st}: artifacts exist, skipped (--resume)")
            summary.append(f"{split}/{st}: skipped (resume)")
            continue
        t0 = time.time()
        with stage(f"{split}/{st}"):
            if st == "prepare":
                stage_prepare(cfg, split)
            elif st == "aliases":
                from .aliases_stage import stage_aliases
                stage_aliases(cfg)
            elif st == "block":
                stage_block(cfg, split)
            elif st == "evaluate":
                from .evaluate import stage_evaluate
                rep = stage_evaluate(cfg)
                summary.append(f"  pair recall {rep['pair_recall']:.4f} | entity recall {rep['entity_recall_mean']} | "
                               f"F0.5 ceiling {rep['f05_ceiling_incl_singletons']:.4f} | cands/S1 {rep['cands_per_s1']['mean']:.1f}")
            elif st == "export":
                from .export import stage_export
                stage_export(cfg, split)
            else:
                raise ValueError(st)
        summary.append(f"{split}/{st}: {fmt_secs(time.time() - t0)}")


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.log_file:
        import sys
        sys.stderr = _Tee(sys.stderr, a.log_file)
    cfg = config_from_args(a)
    splits = ["train", "test"] if a.split == "both" else [a.split]
    wanted = a.stages.split(",") if a.stages else None
    t0 = time.time(); summary: list[str] = []
    log(f"config: {cfg.to_dict()}")
    for split in splits:
        stages = TRAIN_STAGES if split == "train" else TEST_STAGES
        if wanted:
            stages = [s for s in stages if s in wanted]
        run_split(cfg, split, stages, a.resume, summary)
    log("SUMMARY\n  " + "\n  ".join(summary) + f"\n  total {fmt_secs(time.time() - t0)}\n  artifacts: {cfg.artifacts_dir.resolve()}  output: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
