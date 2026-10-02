"""Stage-B trainer (GEMMA_PLAN.md section 4): LoRA + 1-logit answer head over pre-tokenised blocks, DDP via torchrun.

Local (laptop, 4-bit):
    python -m business_entity_resolution.ml.llm.train --train-dir dev/mini_artifacts/train/ml/blocks/train \
        --dev-dir dev/mini_artifacts/train/ml/blocks/dev --model google/gemma-4-E2B --load-4bit \
        --n-blocks 200 --max-steps 150 --eval-every 50 --token-budget 2000
SageMaker: aws/sagemaker_launch.py (channels train/dev/base_model, SM_MODEL_DIR, /opt/ml/checkpoints, torchrun).

Dataset size and instance are both run-time choices:
- `--n-blocks N` trains on the quota-preserving prefix (sample_rank < N) of whatever was uploaded (0 = all of it);
- `--accum auto` keeps ~`--target-blocks-per-step` blocks per optimizer step for any GPU count / token budget;
- `--time-budget-hours H` measures real step time after `--calibrate-steps`, then shrinks the plan (cosine schedule
  included) so training, the final eval and any in-job scoring end inside H, whatever instance the job landed on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .blockio import BatchDataset, BlockMeta, BlockStore, EVAL_COLS, TRAIN_COLS, padded_len, plan_batches, rank_batches
    from .metrics import fit_temperature, summarize_blocks
    from .model import (add_lora, BlockScorer, load_backbone, load_trainable_state, make_loader, predict_logits,
                        resolve_ple_device, save_final, shared_snapshot, to_device, trainable_state)
    from .runtime import (BOOL, all_gather_object, barrier, broadcast_floats, channel_dir, cleanup_distributed,
                          default_checkpoint_dir, env_path, fmt_secs, init_distributed, is_main, list_parquet, load_json, log,
                          pbar, save_json, seed_everything, set_fast_math, stage)
except ImportError:
    from blockio import BatchDataset, BlockMeta, BlockStore, EVAL_COLS, TRAIN_COLS, padded_len, plan_batches, rank_batches
    from metrics import fit_temperature, summarize_blocks
    from model import (add_lora, BlockScorer, load_backbone, load_trainable_state, make_loader, predict_logits,
                       resolve_ple_device, save_final, shared_snapshot, to_device, trainable_state)
    from runtime import (BOOL, all_gather_object, barrier, broadcast_floats, channel_dir, cleanup_distributed,
                         default_checkpoint_dir, env_path, fmt_secs, init_distributed, is_main, list_parquet, load_json, log,
                         pbar, save_json, seed_everything, set_fast_math, stage)

T0 = time.time()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data / paths (SageMaker env vars are the defaults)")
    g.add_argument("--model", default=env_path("SM_CHANNEL_BASE_MODEL", default="google/gemma-4-E4B"),
                   help="base model: local snapshot dir or Hub id")
    g.add_argument("--base-model-id", default="google/gemma-4-E4B", help="Hub id recorded in run_config.json (informational)")
    g.add_argument("--train-dir", default=channel_dir("train"))
    g.add_argument("--dev-dir", default=channel_dir("dev"))
    g.add_argument("--output-dir", default=env_path("SM_MODEL_DIR", default="llm_out/model"))
    g.add_argument("--checkpoint-dir", default=default_checkpoint_dir("llm_out/checkpoints"))
    g = p.add_argument_group("dataset size / schedule")
    g.add_argument("--n-blocks", type=int, default=0, help="train on sample_rank < N (quota-preserving prefix); 0 = all")
    g.add_argument("--dev-blocks", type=int, default=0, help="cap on llm_dev blocks used for eval (0 = all)")
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--max-steps", type=int, default=0, help="exact number of optimizer steps, cycling epochs as needed (0 = use --epochs)")
    g.add_argument("--time-budget-hours", type=float, default=0.0, help="auto-fit the plan to this wall time (0 = off)")
    g.add_argument("--reserve-minutes", type=float, default=20.0, help="kept free at the end for final eval + saving")
    g.add_argument("--calibrate-steps", type=int, default=20, help="optimizer steps measured before the budget re-plan")
    g = p.add_argument_group("batching")
    g.add_argument("--token-budget", type=int, default=5000, help="padded tokens per micro-batch per GPU")
    g.add_argument("--eval-token-budget", type=int, default=0, help="0 = 4 x token-budget")
    g.add_argument("--accum", default="auto", help="gradient accumulation (int or 'auto')")
    g.add_argument("--target-blocks-per-step", type=int, default=64, help="global blocks per optimizer step for --accum auto")
    g.add_argument("--max-blocks-per-batch", type=int, default=64)
    g.add_argument("--shuffle-cands", **BOOL, default=True, help="shuffle candidate order per epoch (order robustness)")
    g.add_argument("--num-workers", type=int, default=-1, help="-1 = auto (0 on Windows, 8 / GPU-share on Linux)")
    g = p.add_argument_group("model")
    g.add_argument("--lora-r", type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=0, help="0 = lora-r")
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--bf16", **BOOL, default=True)
    g.add_argument("--load-4bit", **BOOL, default=False, help="NF4 base weights (laptop debugging only)")
    g.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    g.add_argument("--ple-device", default="auto", choices=["auto", "cpu", "cuda"], help="where the PLE table lives")
    g.add_argument("--grad-ckpt", **BOOL, default=True)
    g.add_argument("--liger", **BOOL, default=False, help="T7: apply liger-kernel patches if they exist for the arch")
    g.add_argument("--compile", **BOOL, default=False, help="T17: torch.compile the backbone (off by default)")
    g = p.add_argument_group("optimisation")
    g.add_argument("--optim", default="paged_adamw_8bit", choices=["paged_adamw_8bit", "adamw"])
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--lr-head", type=float, default=1e-3)
    g.add_argument("--warmup", type=float, default=0.03)
    g.add_argument("--min-lr-ratio", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--betas", default="0.9,0.95")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--spherical-weight", type=float, default=0.0, help="T20: + w x (1 - spherical score) auxiliary loss (off)")
    g = p.add_argument_group("loop")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=1000)
    g.add_argument("--checkpoint-every", type=int, default=1000)
    g.add_argument("--keep-checkpoints", type=int, default=2)
    g.add_argument("--resume", **BOOL, default=False, help="continue from the newest complete checkpoint")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--profile-steps", type=int, default=0, help="torch.profiler over N steps (after 5 warm-up steps)")
    g = p.add_argument_group("optional in-job scoring (uses the time left in the same job)")
    g.add_argument("--score-splits", default="", help="e.g. 'valid,test': blocks from SM_CHANNEL_BLOCKS_<SPLIT> or --score-root")
    g.add_argument("--score-root", default=None, help="local dir with <split>/ block subdirs")
    g.add_argument("--score-out", default=None)
    g.add_argument("--score-token-budget", type=int, default=32000)
    g.add_argument("--score-routed-only", **BOOL, default=False)
    g.add_argument("--score-speedup", type=float, default=3.0, help="scoring blocks/s relative to training (planning only)")
    g.add_argument("--score-budget-fraction", type=float, default=0.4, help="max share of the time budget reserved for in-job scoring")
    return p


# ---------------------------------------------------------------- plan / schedule
@dataclass
class Plan:
    total_steps: int
    warmup_steps: int
    initial_steps: int
    steps_per_epoch: int
    accum: int
    blocks: int


def lr_factor(step: int, plan: Plan, min_ratio: float) -> float:
    if step < plan.warmup_steps:
        return (step + 1) / plan.warmup_steps
    prog = min(1.0, (step - plan.warmup_steps) / max(1, plan.total_steps - plan.warmup_steps))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def spherical_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - spherical score of a Bernoulli forecast (a proper scoring rule; T20)."""
    p = torch.sigmoid(logits)
    return (1 - torch.where(y > 0.5, p, 1 - p) / torch.sqrt(p ** 2 + (1 - p) ** 2)).mean()


def make_optimizer(kind: str, groups: list[dict], betas):
    if kind == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
            probe = torch.nn.Parameter(torch.zeros(64, device=groups[0]["params"][0].device))
            probe.grad = torch.zeros_like(probe)
            bnb.optim.PagedAdamW8bit([probe], lr=1e-3).step()   # a wheel/CUDA mismatch surfaces here, not at step 1
            return bnb.optim.PagedAdamW8bit(groups, betas=betas)   # T6
        except Exception as e:  # documented fallback (+~3 GB of fp32 optimizer state)
            log(f"bitsandbytes 8-bit optimizer unavailable ({type(e).__name__}: {e}); falling back to torch AdamW")
    return torch.optim.AdamW(groups, betas=betas, fused=torch.cuda.is_available())


# ---------------------------------------------------------------- checkpoints
def rng_state(device) -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        s["cuda"] = torch.cuda.get_rng_state(device)
    return s


def set_rng_state(s: dict, device) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if device.type == "cuda" and "cuda" in s:
        torch.cuda.set_rng_state(s["cuda"], device)


def newest_checkpoint(root: Path) -> Path | None:
    done = sorted(d for d in root.glob("step_*") if (d / "COMPLETE").exists())
    return done[-1] if done else None


def save_checkpoint(root: Path, step: int, scorer, opt, sched, state: dict, device, keep: int) -> None:
    d = root / f"step_{step:07d}"
    if is_main():
        d.mkdir(parents=True, exist_ok=True)
    barrier()
    torch.save(rng_state(device), d / f"rng_rank{int(os.environ.get('RANK', 0))}.pt")
    barrier()
    if is_main():
        torch.save(trainable_state(scorer), d / "trainable.pt")
        torch.save(opt.state_dict(), d / "optimizer.pt")
        torch.save(sched.state_dict(), d / "scheduler.pt")
        save_json(state, d / "state.json")
        (d / "COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        for old in sorted(x for x in root.glob("step_*") if (x / "COMPLETE").exists())[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
        log(f"checkpoint step {step} -> {d}")
    barrier()


# ---------------------------------------------------------------- eval
def run_eval(scorer, dev: BlockStore, a, device, rank: int, world: int, temperature: float = 1.0):
    """Every rank scores a strided shard of llm_dev; rank 0 assembles and summarises. Returns (metrics, flat logits)."""
    local = np.arange(rank, len(dev), world)
    lg = predict_logits(scorer, dev, local, token_budget=a.eval_token_budget, device=device, desc="eval llm_dev")
    gathered = all_gather_object((local, lg))
    if not is_main():
        return None, None
    per_block = [None] * len(dev)
    for ids, lgs in gathered:
        for i, x in zip(ids, lgs):
            per_block[int(i)] = x
    flat = np.concatenate(per_block)
    country = dev.col("country")
    masks = {}
    if country is not None:
        masks.update(india=country == "India", us=country == "US")
    for c in ("has_indic", "is_hard"):
        if dev.col(c) is not None:
            masks[c.replace("has_", "").replace("is_", "")] = dev.col(c).astype(bool)
    m = summarize_blocks(flat, dev.labels, dev.m_off, dev.col("n_truth"), p_base=dev.p_gbm, masks=masks, temperature=temperature)
    return m, flat


def fmt_metrics(m: dict) -> str:
    keys = ["bce", "auc", "ece", "f05", "f05_threshold", "f05_base", "f05_india", "f05_us", "f05_indic", "f05_base_indic", "f05_hard"]
    return " | ".join(f"{k} {m[k]:.4f}" for k in keys if k in m and m[k] == m[k])


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    rank, local_rank, world, device = init_distributed()
    set_fast_math()
    seed_everything(a.seed + rank)
    if not a.train_dir:
        raise SystemExit("--train-dir (or SM_CHANNEL_TRAIN) is required")
    a.eval_token_budget = a.eval_token_budget or 4 * a.token_budget
    a.lora_alpha = a.lora_alpha or a.lora_r
    betas = tuple(float(x) for x in a.betas.split(","))
    if a.num_workers < 0:
        a.num_workers = 0 if os.name == "nt" else max(1, min(4, ((os.cpu_count() or 8) - 2) // max(1, world)))
    ckpt_root, out_dir = Path(a.checkpoint_dir), Path(a.output_dir)
    log(f"args: {json.dumps(vars(a), default=str)}")
    log(f"world={world} device={device} torch={torch.__version__} "
        f"sdpa flash={torch.backends.cuda.flash_sdp_enabled()} mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}")

    # ---------------- data (pre-tokenised; nothing is tokenised here, T8)
    with stage("load blocks"):
        meta = BlockMeta.load(a.train_dir)
        train = BlockStore.load(a.train_dir, columns=TRAIN_COLS, n_blocks=a.n_blocks)
        dev = BlockStore.load(a.dev_dir, columns=EVAL_COLS, limit=a.dev_blocks, meta=meta) if a.dev_dir else None
        if train.labels is None:
            raise SystemExit("training blocks carry no labels")
        mean_tok = float(train.n_tokens.mean())
        est_bpm = max(1.0, a.token_budget / padded_len(int(mean_tok * 1.05)))
        accum = int(a.accum) if str(a.accum) != "auto" else max(1, round(a.target_blocks_per_step / (world * est_bpm)))
        ep0 = rank_batches(plan_batches(train.n_tokens, a.token_budget, seed=a.seed, epoch=0, max_blocks=a.max_blocks_per_batch), rank, world, accum)
        if not ep0:
            raise SystemExit(f"{len(train)} blocks are too few for world={world} x accum={accum}")
        spe = len(ep0) // accum
        total = a.max_steps if a.max_steps > 0 else max(1, math.ceil(a.epochs * spe))
        plan = Plan(total_steps=total, warmup_steps=max(1, round(a.warmup * total)), initial_steps=total,
                    steps_per_epoch=spe, accum=accum, blocks=len(train))
        pad = 1 - train.n_tokens[np.concatenate(ep0)].sum() / sum(len(b) * padded_len(int(train.n_tokens[b].max())) for b in ep0)
        log(f"train blocks {len(train):,} (requested n_blocks={a.n_blocks or 'all'}) | tokens/block p50 {np.median(train.n_tokens):.0f} "
            f"p95 {np.percentile(train.n_tokens, 95):.0f} | label rate {train.labels.mean():.3f} | micro-batches/rank/epoch {len(ep0)} "
            f"(~{len(np.concatenate(ep0)) / len(ep0):.1f} blocks, padding {100 * pad:.1f}%) | accum {accum} | "
            f"~{a.target_blocks_per_step if str(a.accum) == 'auto' else '?'} blocks/step | steps/epoch {spe} | planned steps {total}")

    # ---------------- model
    with stage("load model"):
        a.model = str(shared_snapshot(a.model, device))   # Hub id: rank 0 downloads once, the other ranks wait
        amp = torch.bfloat16 if a.bf16 else torch.float32
        ple_dev = resolve_ple_device(a.ple_device, device, training=True)
        backbone, ple, tcfg = load_backbone(a.model, device=device, dtype=amp, load_4bit=a.load_4bit, attn=a.attn,
                                            ple_device=ple_dev, grad_ckpt=a.grad_ckpt, liger=a.liger)
        backbone = add_lora(backbone, a.lora_r, a.lora_alpha, a.lora_dropout)
        scorer = BlockScorer(backbone, tcfg.hidden_size, ple)
        scorer.head.to(device)
        if a.compile:
            scorer.set_compiled(torch.compile(scorer.backbone, dynamic=True))   # T17: forward only; module tree unchanged
        lora_params = [p for n, p in scorer.backbone.named_parameters() if p.requires_grad]
        head_params = list(scorer.head.parameters())
        opt = make_optimizer(a.optim, [dict(params=lora_params, lr=a.lr, weight_decay=a.weight_decay),
                                       dict(params=head_params, lr=a.lr_head, weight_decay=0.0)], betas)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, plan, a.min_lr_ratio))
        trainable = lora_params + head_params
        if device.type == "cuda":
            log(f"GPU memory after load: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB allocated")

    # ---------------- resume (T16)
    step, epoch, micro, history = 0, 0, 0, []
    ckpt = newest_checkpoint(ckpt_root) if a.resume else None
    if ckpt is not None:
        with stage(f"resume from {ckpt}"):
            st = load_json(ckpt / "state.json")
            load_trainable_state(scorer, torch.load(ckpt / "trainable.pt", map_location="cpu"))
            opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=False))
            step, epoch = st["step"], st["epoch"]
            plan.total_steps, plan.warmup_steps = st["plan"]["total_steps"], st["plan"]["warmup_steps"]
            if st.get("world") == world and st.get("accum") == accum:
                micro = st["micro"]
            else:  # resumed on a different instance: convert the data cursor via global micro-batches
                micro = (st["micro"] * st.get("world", world) // world) // accum * accum
                log(f"world/accum changed ({st.get('world')}/{st.get('accum')} -> {world}/{accum}); data cursor mapped to micro {micro}")
            sched.load_state_dict(torch.load(ckpt / "scheduler.pt", map_location="cpu", weights_only=False))
            rf = ckpt / f"rng_rank{rank}.pt"
            if rf.exists():
                set_rng_state(torch.load(rf, map_location="cpu", weights_only=False), device)
            history = st.get("history", [])
            log(f"resumed at step {step}/{plan.total_steps}, epoch {epoch}, micro-batch {micro}")

    # ---------------- DDP (T11)
    model = scorer
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        frozen = [n for n, p in scorer.named_parameters() if not p.requires_grad] + [n for n, _ in scorer.named_buffers()]
        DDP._set_params_and_buffers_to_ignore_for_model(scorer, frozen)  # identical on every rank: no 15 GB broadcast
        on_gpu = device.type == "cuda"
        model = DDP(scorer, device_ids=[local_rank] if on_gpu else None, output_device=local_rank if on_gpu else None,
                    find_unused_parameters=False, gradient_as_bucket_view=True, bucket_cap_mb=100, broadcast_buffers=False)

    # ---------------- planning helpers for the time budget
    budget_s = a.time_budget_hours * 3600
    reserve_s = a.reserve_minutes * 60
    if budget_s and reserve_s >= 0.5 * budget_s:
        log(f"WARNING: --reserve-minutes {a.reserve_minutes:g} is >= half of the {a.time_budget_hours:g} h budget; reserve cut to 10%")
        reserve_s = 0.1 * budget_s
    score_splits = [s for s in a.score_splits.split(",") if s]
    n_score = _count_score_blocks(a, score_splits) if score_splits else 0
    n_evals_left = lambda: (max(0, (plan.total_steps - step) // a.eval_every) if a.eval_every and dev else 0) + (1 if dev else 0)
    eval_s = None
    step_times: list[float] = []
    blocks_per_step_local: list[int] = []

    def replan(reason: str) -> bool:
        if not budget_s or len(step_times) < 3:
            return False
        spd = float(np.median(step_times[-50:]))
        gbps = world * float(np.mean(blocks_per_step_local[-50:])) / spd       # global training blocks/s
        e_s = eval_s if eval_s is not None else (len(dev) / (a.score_speedup * gbps) if dev else 0.0)
        s_s = n_score / (a.score_speedup * gbps) if n_score else 0.0
        if s_s > a.score_budget_fraction * budget_s:
            # scoring is resumable (per-shard parts), training is not: never let the scoring reserve eat the run
            log(f"WARNING: in-job scoring of {n_score:,} blocks needs ~{fmt_secs(s_s)} at ~{a.score_speedup * gbps:.1f} blocks/s, "
                f"more than {a.score_budget_fraction:.0%} of the budget; reserving only that share - scoring will be cut by max_run "
                f"and must be finished by a separate `score` job (same --run-name resumes the parts)")
            s_s = a.score_budget_fraction * budget_s
        left = budget_s - (time.time() - T0) - reserve_s - s_s - e_s * n_evals_left()
        afford = step + int(left / spd)
        if afford < plan.total_steps * 0.98:
            new = max(step + 1, afford)
            log(f"time budget ({reason}): {spd:.2f} s/step, {gbps:.1f} blocks/s -> plan {plan.total_steps} -> {new} steps "
                f"(~{new * world * float(np.mean(blocks_per_step_local[-50:])):,.0f} blocks); "
                f"reserved: eval {fmt_secs(e_s * n_evals_left())}, scoring {fmt_secs(s_s)} ({n_score:,} blocks), final {fmt_secs(reserve_s)}")
            plan.total_steps = new
        else:
            log(f"time budget ({reason}): {spd:.2f} s/step, {gbps:.1f} blocks/s -> {plan.total_steps} steps fit "
                f"(ETA {fmt_secs((plan.total_steps - step) * spd)}, budget left {fmt_secs(left)})")
        return True

    # ---------------- train loop
    log_path = out_dir / "train_log.jsonl"
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    scorer.train()
    bar = pbar(total=plan.total_steps, desc="train", unit="step", initial=step)
    win = dict(loss=torch.zeros((), device=device), n=0, tok=0, pad=0, blk=0, t=time.time(), gn=0.0, pause=0.0)
    last_end, stop, calibrated, prof, blocks_seen = time.time(), False, False, None, 0
    while not stop and step < plan.total_steps:
        batches = rank_batches(plan_batches(train.n_tokens, a.token_budget, seed=a.seed, epoch=epoch, max_blocks=a.max_blocks_per_batch),
                               rank, world, accum)
        ds = BatchDataset(train, batches[micro:], shuffle_cands=a.shuffle_cands, seed=a.seed, epoch=epoch)
        blocks_in_step = 0
        for j, batch in enumerate(make_loader(ds, a.num_workers, device.type == "cuda"), start=micro):
            last = (j + 1) % accum == 0
            inputs = to_device(batch, device)
            labels = batch["labels"].to(device, non_blocking=True)
            with (model.no_sync() if (world > 1 and not last) else nullcontext()):
                with torch.autocast(device_type=device.type, dtype=amp, enabled=a.bf16):
                    logits = model(**inputs)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
                if a.spherical_weight:
                    loss = loss + a.spherical_weight * spherical_loss(logits.float(), labels)
                (loss / accum).backward()
            win["loss"] += loss.detach()
            win["n"] += 1
            win["tok"] += batch["n_real_tokens"]
            win["pad"] += batch["n_padded_tokens"]
            win["blk"] += len(batch["block_index"])
            blocks_in_step += len(batch["block_index"])
            blocks_seen += len(batch["block_index"])
            if not last:
                continue
            gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step, micro = step + 1, j + 1
            now = time.time()
            if step > 3:
                step_times.append(now - last_end)
                blocks_per_step_local.append(blocks_in_step)
            last_end, blocks_in_step = now, 0
            win["gn"] = float(gn)
            bar.update(1)

            if a.profile_steps and step == 5:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
            if prof is not None and step == 5 + a.profile_steps:
                prof.__exit__(None, None, None)
                if is_main():
                    tbl = prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=15)
                    attn = sorted({e.key for e in prof.key_averages() if any(s in e.key.lower() for s in ("flash", "efficient_attention", "cudnn_attention", "sdpa"))})
                    log(f"profile ({a.profile_steps} steps):\n{tbl}\nattention kernels: {attn}")
                prof = None

            # time budget: re-plan once after calibration (and after every eval); hard stop if over budget
            if budget_s and is_main():
                if not calibrated and step >= min(a.calibrate_steps, max(4, plan.total_steps // 2)):
                    calibrated = replan(f"calibration after {step} steps")
                if time.time() - T0 > budget_s - reserve_s:
                    log("time budget exhausted: stopping now")
                    stop = True
            ctrl = broadcast_floats([1.0 if stop else 0.0, float(plan.total_steps)], device) if budget_s else [0.0, plan.total_steps]
            stop, plan.total_steps = bool(ctrl[0]), int(ctrl[1])
            bar.total = plan.total_steps

            if step % a.log_every == 0 or step == 1 or step == plan.total_steps:
                dt = max(1e-6, time.time() - win["t"] - win["pause"])   # eval / checkpoint time is not training speed
                lr0, lr1 = sched.get_last_lr()[:2]
                spd = float(np.median(step_times[-50:])) if step_times else dt
                rec = {"step": step, "epoch": epoch, "loss": float(win["loss"]) / max(1, win["n"]), "lr": lr0, "lr_head": lr1,
                       "blocks_per_s": world * win["blk"] / dt, "tokens_per_s": world * win["tok"] / dt,
                       "padding": 1 - win["tok"] / max(1, win["pad"]), "grad_norm": win["gn"],
                       "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
                       "elapsed_s": time.time() - T0, "eta_s": (plan.total_steps - step) * spd}
                log(f"step {step}/{plan.total_steps} | loss {rec['loss']:.4f} | lr {lr0:.2e}/{lr1:.2e} | {rec['blocks_per_s']:.1f} blk/s "
                    f"{rec['tokens_per_s'] / 1e3:.1f}k tok/s | pad {100 * rec['padding']:.1f}% | gn {rec['grad_norm']:.2f} | "
                    f"mem {rec['peak_mem_gb']:.1f} GB | elapsed {fmt_secs(rec['elapsed_s'])} | ETA {fmt_secs(rec['eta_s'])}")
                if is_main():
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                history.append(rec)
                win = dict(loss=torch.zeros((), device=device), n=0, tok=0, pad=0, blk=0, t=time.time(), gn=win["gn"], pause=0.0)

            if dev is not None and a.eval_every and step % a.eval_every == 0 and step < plan.total_steps:
                t_ev = time.time()
                m, _ = run_eval(scorer, dev, a, device, rank, world)
                eval_s = time.time() - t_ev
                if is_main():
                    log(f"eval @ step {step}: {fmt_metrics(m)} ({fmt_secs(eval_s)})")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "eval": m}) + "\n")
                    history.append({"step": step, "eval": m})
                    if budget_s:
                        replan(f"after eval @ {step}")
                ctrl = broadcast_floats([0.0, float(plan.total_steps)], device)
                plan.total_steps = int(ctrl[1])
                win["pause"] += time.time() - t_ev
                last_end = time.time()
            if a.checkpoint_every and step % a.checkpoint_every == 0 and step < plan.total_steps:
                state = {"step": step, "epoch": epoch, "micro": micro, "world": world, "accum": accum, "plan": asdict(plan),
                         "history": history[-200:], "args": vars(a)}
                t_ck = time.time()
                save_checkpoint(ckpt_root, step, scorer, opt, sched, state, device, a.keep_checkpoints)
                win["pause"] += time.time() - t_ck
                last_end = time.time()
            if stop or step >= plan.total_steps:
                break
        else:
            epoch, micro = epoch + 1, 0
            continue
        break
    bar.close()
    train_s = time.time() - T0
    log(f"training finished: {step} steps, {blocks_seen * world:,} blocks seen this run ({blocks_seen * world / max(1, len(train)):.2f} epochs), "
        f"{fmt_secs(train_s)} since start")

    # ---------------- final eval, temperature scaling (T19), save
    barrier()
    temperature, final = 1.0, {}
    if dev is not None:
        with stage("final llm_dev eval + temperature scaling"):
            m, flat = run_eval(scorer, dev, a, device, rank, world)
            if is_main():
                temperature = fit_temperature(flat, dev.labels)
                m_cal = summarize_blocks(flat, dev.labels, dev.m_off, dev.col("n_truth"), p_base=dev.p_gbm, temperature=temperature)
                final = {"raw": m, "calibrated": m_cal, "temperature": temperature}
                log(f"final llm_dev: {fmt_metrics(m)}\ntemperature {temperature:.3f}: ece {m['ece']:.4f} -> {m_cal['ece']:.4f}, "
                    f"bce {m['bce']:.4f} -> {m_cal['bce']:.4f}")
    temperature = broadcast_floats([temperature], device)[0]
    if is_main():
        run_config = {"base_model": a.model, "base_model_id": a.base_model_id, "lora": {"r": a.lora_r, "alpha": a.lora_alpha,
                      "dropout": a.lora_dropout}, "load_4bit": a.load_4bit, "attn": a.attn, "blocks_meta": meta.to_dict(),
                      "n_blocks": len(train), "steps": step, "planned_steps": plan.initial_steps, "accum": accum, "world": world,
                      "token_budget": a.token_budget, "train_seconds": train_s, "final_dev": final, "args": vars(a)}
        for d in (out_dir, ckpt_root / "final"):
            save_final(scorer, d, run_config, temperature)
            save_json(final, d / "dev_metrics.json")
            save_json(meta.to_dict(), d / "blocks_meta.json")
        if log_path.exists():
            shutil.copyfile(log_path, ckpt_root / "final" / "train_log.jsonl")
    barrier()

    # ---------------- optional scoring in the same job
    if score_splits:
        try:
            from .score import score_splits as run_scoring
        except ImportError:
            from score import score_splits as run_scoring
        del model
        if not a.load_4bit and hasattr(scorer.backbone, "merge_and_unload"):
            scorer.backbone = scorer.backbone.merge_and_unload()
        scorer.eval()
        run_scoring(scorer, temperature, score_splits, root=a.score_root, out_dir=a.score_out, final_dir=a.output_dir,
                    token_budget=a.score_token_budget, routed_only=a.score_routed_only, device=device, rank=rank, world=world,
                    num_workers=a.num_workers)
    cleanup_distributed()
    log(f"done in {fmt_secs(time.time() - T0)}")


def _count_score_blocks(a, splits: list[str]) -> int:
    import pyarrow.parquet as pq
    try:
        from .score import split_blocks_dir
    except ImportError:
        from score import split_blocks_dir
    n = 0
    for s in splits:
        d = split_blocks_dir(s, a.score_root)
        if not d:
            continue
        meta = BlockMeta.load(d).extra
        if a.score_routed_only and s == "test" and all("n_routed" in f for f in meta.get("files", [])):
            n += sum(int(f["n_routed"]) for f in meta["files"])
        else:
            n += sum(pq.ParquetFile(f).metadata.num_rows for f in list_parquet(d))
    return n


if __name__ == "__main__":
    main(sys.argv[1:])
