"""Stage-B scorer (GEMMA_PLAN.md section 6): merged-LoRA bf16 inference over pre-tokenised blocks.

    python -m business_entity_resolution.ml.llm.score --model-dir llm_out/model --blocks-root dev/mini_artifacts/train/ml/blocks \
        --splits valid --final-dir dev/mini_artifacts/train/ml
SageMaker (launched by aws/sagemaker_launch.py score): channels `model` (the training job's model.tar.gz, extracted
here), `base_model`, `blocks_<split>`; parts go to /opt/ml/checkpoints/scores (synced to S3 continuously, so a re-run
resumes), the merged `llm_scores_<split>.parquet` to SM_MODEL_DIR.

Output columns: s1_id, cand_id, cand_rank (position in the block, stage-A order), p_gbm, logit[, logit_rev, p_fwd, p_rev], p_llm
(= sigmoid(logit / T) with T from llm_dev temperature scaling; the mean of both orders with --two-order).

Uncertainty routing (CATBOOST_PLAN.md section 5): `--entity-list test/ml/gbm_entity_test.parquet --top-k K` scores only
the first K test entities of that file's order (route_rank: France first if the France check flagged it, then CatBoost's
u = sum min(p, 1-p) descending). K >= 1 is a count, 0 < K < 1 a share, 0 = all. valid/dev are always scored fully.
On SageMaker the list arrives as the `entity_list` channel.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from .blockio import BlockMeta, BlockStore, SCORE_COLS
    from .metrics import sigmoid
    from .model import load_trained, predict_logits, resolve_ple_device, shared_snapshot
    from .runtime import (BOOL, barrier, channel_dir, cleanup_distributed, default_checkpoint_dir, env_path, fmt_secs,
                          init_distributed, is_main, list_parquet, load_json, log, resolve_model_artifact, set_fast_math, stage)
except ImportError:
    from blockio import BlockMeta, BlockStore, SCORE_COLS
    from metrics import sigmoid
    from model import load_trained, predict_logits, resolve_ple_device, shared_snapshot
    from runtime import (BOOL, barrier, channel_dir, cleanup_distributed, default_checkpoint_dir, env_path, fmt_secs,
                         init_distributed, is_main, list_parquet, load_json, log, resolve_model_artifact, set_fast_math, stage)

T0 = time.time()


def split_blocks_dir(split: str, root: str | None = None) -> str | None:
    d = channel_dir(f"blocks_{split}")
    if d:
        return d
    if root and (Path(root) / split).exists():
        return str(Path(root) / split)
    return None


def _part_name(split: str, fi: int, world: int, rank: int) -> str:
    return f"llm_scores_{split}_f{fi:04d}_w{world}_r{rank}.parquet"


def _complete_parts(d: Path, split: str) -> dict[int, list[Path]]:
    """file index -> the part files of a complete (all ranks present) set, whatever world size wrote it."""
    sets: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for f in d.glob(f"llm_scores_{split}_f*_w*_r*.parquet"):
        m = re.match(rf"llm_scores_{split}_f(\d+)_w(\d+)_r(\d+)\.parquet$", f.name)
        if m:
            sets[(int(m[1]), int(m[2]))].append(f)
    done = {}
    for (fi, w), fs in sets.items():
        if len(fs) == w and fi not in done:
            done[fi] = sorted(fs)
    return done


def _write_parquet(cols: dict, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, path)


def apply_fp8(model) -> None:
    """S6: FP8 weight-only quantisation (torchao) for Ada/Hopper tensor cores; off by default, validate first."""
    try:
        from torchao.quantization import quantize_
        try:
            from torchao.quantization import Float8WeightOnlyConfig
            cfg = Float8WeightOnlyConfig()
        except ImportError:
            from torchao.quantization import float8_weight_only
            cfg = float8_weight_only()
        quantize_(model, cfg)
        log("FP8 weight-only quantisation applied (S6)")
    except Exception as e:
        log(f"FP8 unavailable ({type(e).__name__}: {e}); scoring in bf16")


def load_entity_list(path: str | None, top_k: float):
    """s1_ids (pyarrow array) of the first K entities of a gbm_entity_<split>.parquet routing order, or None."""
    if not path:
        return None
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    p = Path(path)
    if p.is_dir():   # a SageMaker channel directory: the single parquet file inside
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet file in {p}")
        p = files[0]
    t = pq.read_table(p)
    order = "route_rank" if "route_rank" in t.column_names else "u_rank"
    t = t.take(pc.sort_indices(t, sort_keys=[(order, "ascending")]))
    n = t.num_rows if top_k <= 0 else (int(round(top_k * t.num_rows)) if top_k < 1 else min(int(top_k), t.num_rows))
    log(f"entity list {p}: routing the first {n:,} of {t.num_rows:,} entities by {order} (--top-k {top_k})")
    return t["s1_id"].slice(0, n).combine_chunks()


def score_splits(scorer, temperature: float, splits: list[str], *, root: str | None, out_dir: str | None, final_dir: str | None,
                 token_budget: int, routed_only: bool, device, rank: int, world: int, num_workers: int = 0,
                 two_order: bool = False, limit: int = 0, overwrite: bool = False, keep_ids=None,
                 keep_splits: tuple[str, ...] = ("test",)) -> dict[str, Path]:
    """Score every requested split; per (shard file, rank) parts make the run resumable (S4). Returns merged files."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(out_dir or (default_checkpoint_dir("llm_out") + "/scores"))
    merged = {}
    for split in splits:
        bdir = split_blocks_dir(split, root)
        if not bdir:
            log(f"[{split}] no blocks channel/dir found: skipped")
            continue
        d = out / split
        if is_main():
            d.mkdir(parents=True, exist_ok=True)
        barrier()
        files = list_parquet(bdir)
        meta = BlockMeta.load(bdir)
        done = {} if overwrite else _complete_parts(d, split)
        routed = routed_only and split == "test"   # valid/dev are always scored fully (the rule is tuned on all of valid_llm)
        ids = keep_ids if split in keep_splits else None
        how = " (routed only)" if routed else ""
        how += f" ({len(ids):,} entities from the entity list)" if ids is not None else ""
        with stage(f"score {split}{how}: {len(files)} shard file(s) from {bdir}, {len(done)} already done"):
            n_seen = 0
            for fi, f in enumerate(files):
                if fi in done:
                    continue
                part = d / _part_name(split, fi, world, rank)
                if part.exists() and not overwrite:
                    continue
                try:
                    store = BlockStore.load(f, columns=SCORE_COLS, shard=(rank, world), routed_only=routed, meta=meta,
                                            limit=max(0, limit - n_seen) if limit else 0, quiet=True, keep_ids=ids)
                except ValueError:   # nothing for this rank in this file (routing / limit): write an empty part
                    _write_parquet({"s1_id": pa.array([], pa.string()), "cand_id": pa.array([], pa.string()),
                                    "cand_rank": pa.array([], pa.int16()), "p_gbm": pa.array([], pa.float32()),
                                    "logit": pa.array([], pa.float32()), "p_llm": pa.array([], pa.float32())}, part)
                    continue
                n_seen += len(store)
                lg = np.concatenate(predict_logits(scorer, store, token_budget=token_budget, device=device, num_workers=num_workers,
                                                   desc=f"{split} {fi + 1}/{len(files)}"))
                n_c = np.diff(store.m_off)
                cols = {"s1_id": np.repeat(store.col("s1_id"), n_c).astype(str), "cand_id": store.cand_ids.astype(str),
                        "cand_rank": (np.arange(len(lg)) - np.repeat(store.m_off[:-1], n_c)).astype(np.int16),
                        "p_gbm": (store.p_gbm if store.p_gbm is not None else np.full(len(lg), np.nan)).astype(np.float32),
                        "logit": lg.astype(np.float32)}
                used = lg
                if two_order:   # S9: reversed candidate order, averaged in logit space
                    rev = np.concatenate(predict_logits(scorer, store, token_budget=token_budget, device=device, reverse=True,
                                                        num_workers=num_workers, desc=f"{split} {fi + 1}/{len(files)} rev"))
                    cols["logit_rev"] = rev.astype(np.float32)
                    cols["p_fwd"] = sigmoid(lg / temperature).astype(np.float32)
                    cols["p_rev"] = sigmoid(rev / temperature).astype(np.float32)
                    used = (lg + rev) / 2
                cols["p_llm"] = sigmoid(used / temperature).astype(np.float32)
                _write_parquet(cols, part)
                if limit and n_seen >= limit:
                    break
        barrier()
        if is_main():
            parts = [p for _, ps in sorted(_complete_parts(d, split).items()) for p in ps]
            tbl = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="default") if parts else None
            if tbl is not None:
                for target in {Path(final_dir) if final_dir else d, d}:
                    target.mkdir(parents=True, exist_ok=True)
                    pq.write_table(tbl, target / f"llm_scores_{split}.parquet")
                merged[split] = (Path(final_dir) if final_dir else d) / f"llm_scores_{split}.parquet"
                p = tbl["p_llm"].to_numpy()
                log(f"[{split}] {tbl.num_rows:,} candidate scores from {len(parts)} parts -> {merged[split]} "
                    f"(p_llm mean {p.mean():.3f}, share >= 0.5: {(p >= 0.5).mean():.3f})")
        barrier()
    return merged


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=channel_dir("model") or "llm_out/model", help="training output dir or model.tar.gz")
    p.add_argument("--model", default=channel_dir("base_model"), help="base model (default: SM channel, else run_config)")
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--blocks-root", default=None, help="local dir with <split>/ block subdirs (SageMaker: blocks_<split> channels)")
    p.add_argument("--out-dir", default=None, help="parts dir (default /opt/ml/checkpoints/scores on SageMaker, llm_out/scores locally)")
    p.add_argument("--final-dir", default=env_path("SM_MODEL_DIR", default=None), help="where llm_scores_<split>.parquet is written")
    p.add_argument("--token-budget", type=int, default=32000, help="padded tokens per batch per GPU (S3)")
    p.add_argument("--ple-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--load-4bit", **BOOL, default=False)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--merge", **BOOL, default=True, help="merge LoRA into the weights (S1)")
    p.add_argument("--two-order", **BOOL, default=False, help="S9: also score the reversed candidate order and average")
    p.add_argument("--routed-only", **BOOL, default=False, help="S5: only router-flagged blocks of the test split")
    p.add_argument("--entity-list", default=channel_dir("entity_list"),
                   help="gbm_entity_test.parquet (CatBoost routing order); default: the SageMaker entity_list channel")
    p.add_argument("--top-k", type=float, default=0.0, help="score the first K entities of --entity-list (count, or share if < 1; 0 = all)")
    p.add_argument("--fp8", **BOOL, default=False, help="S6: torchao FP8 weight-only (validate against bf16 first)")
    p.add_argument("--limit", type=int, default=0, help="score at most N blocks per split per rank (smoke tests)")
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--overwrite", **BOOL, default=False)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    rank, _, world, device = init_distributed()
    set_fast_math()
    if a.num_workers < 0:
        a.num_workers = 0 if os.name == "nt" else 2
    with stage("load trained scorer"):
        model_dir = resolve_model_artifact(a.model_dir)
        if not a.model:   # no base_model channel: the Hub id recorded at training time (the recorded path was job-local)
            run0 = load_json(model_dir / "run_config.json")
            a.model = run0["base_model"] if Path(run0["base_model"]).exists() else run0.get("base_model_id", run0["base_model"])
        a.model = str(shared_snapshot(a.model, device))   # rank 0 downloads, the others wait
        scorer, run, temperature = load_trained(model_dir, base_model=a.model, device=device, load_4bit=a.load_4bit, attn=a.attn,
                                                ple_device=resolve_ple_device(a.ple_device, device, training=False), merge=a.merge)
        if a.fp8:
            apply_fp8(scorer.backbone)
        log(f"model {model_dir} (trained on {run.get('n_blocks')} blocks, {run.get('steps')} steps), temperature {temperature:.3f}")
    keep_ids = load_entity_list(a.entity_list, a.top_k)
    if keep_ids is not None and a.routed_only:
        log("both --entity-list and --routed-only given: a test block is scored only if it satisfies both")
    with torch.inference_mode():
        score_splits(scorer, temperature, [s for s in a.splits.split(",") if s], root=a.blocks_root, out_dir=a.out_dir,
                     final_dir=a.final_dir, token_budget=a.token_budget, routed_only=a.routed_only, device=device, rank=rank,
                     world=world, num_workers=a.num_workers, two_order=a.two_order, limit=a.limit, overwrite=a.overwrite,
                     keep_ids=keep_ids)
    cleanup_distributed()
    log(f"done in {fmt_secs(time.time() - T0)}")


if __name__ == "__main__":
    main(sys.argv[1:])
