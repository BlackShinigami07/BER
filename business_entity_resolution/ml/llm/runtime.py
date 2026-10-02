"""Runtime helpers for train.py / score.py: SageMaker paths, torch.distributed, logging, progress bars.

This folder (ml/llm/) is the SageMaker `source_dir`: it is copied into the job container on its own, so nothing
here may import from the parent `business_entity_resolution` package. Local runs import it as a package.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np

IS_TTY = sys.stderr.isatty()
ON_SAGEMAKER = bool(os.environ.get("SM_TRAINING_ENV") or os.environ.get("SM_CURRENT_HOST") or Path("/opt/ml/input/config").exists())
SM_CHECKPOINT_DIR = "/opt/ml/checkpoints"


# ---------------------------------------------------------------- argument helpers
def str2bool(v) -> bool:
    """argparse type for booleans. SageMaker passes hyperparameters as `--flag true`, so plain store_true is not enough."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", "none", ""):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


BOOL = dict(type=str2bool, nargs="?", const=True)


def env_path(*names: str, default: str | None = None) -> str | None:
    """First set environment variable among `names` (SageMaker channel / model dirs), else `default`."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def channel_dir(name: str) -> str | None:
    """SageMaker input channel directory (`SM_CHANNEL_<NAME>`), None when not on SageMaker or not wired."""
    return os.environ.get(f"SM_CHANNEL_{name.upper()}")


# ---------------------------------------------------------------- distributed
def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return rank() == 0


def init_distributed():
    """torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; one process per GPU. Returns (rank, local_rank, world, device)."""
    import torch
    r, lr, w = rank(), local_rank(), world()
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
        device = torch.device("cuda", lr)
    else:
        device = torch.device("cpu")
    if w > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                             timeout=timedelta(minutes=60), device_id=device if device.type == "cuda" else None)
    return r, lr, w, device


def barrier() -> None:
    import torch
    if world() > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_gather_object(obj) -> list:
    import torch
    if world() == 1 or not torch.distributed.is_initialized():
        return [obj]
    out = [None] * world()
    torch.distributed.all_gather_object(out, obj)
    return out


def broadcast_floats(values: list[float], device) -> list[float]:
    """Rank 0's values on every rank (one tiny collective; used for the stop flag / re-planned step count)."""
    import torch
    if world() == 1 or not torch.distributed.is_initialized():
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.broadcast(t, src=0)
    return t.tolist()


def cleanup_distributed() -> None:
    import torch
    if world() > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------- logging / progress
def fmt_secs(s: float) -> str:
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    return f"{h}h{m:02d}m{r:02d}s" if h else (f"{m}m{r:02d}s" if m else f"{r}s")


def log(msg: str, all_ranks: bool = False) -> None:
    if not (all_ranks or is_main()):
        return
    line = f"[{time.strftime('%H:%M:%S')}]{f' [rank {rank()}]' if world() > 1 else ''} {msg}"
    try:
        from tqdm import tqdm
        tqdm.write(line, file=sys.stderr)
    except ImportError:
        print(line, file=sys.stderr, flush=True)


def pbar(total: int | None = None, desc: str = "", unit: str = "it", **kw):
    """tqdm on rank 0 only; refresh every 30 s when stderr is a file (CloudWatch stays readable)."""
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit=unit, file=sys.stderr, dynamic_ncols=True, disable=not is_main(),
                mininterval=0.5 if IS_TTY else 30.0, smoothing=0.05, **kw)


@contextmanager
def stage(name: str):
    log(f"{'=' * 70}\n>>> {name}")
    t0 = time.time()
    try:
        yield
    finally:
        log(f"<<< {name} done in {fmt_secs(time.time() - t0)}")


# ---------------------------------------------------------------- reproducibility / numerics
def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_fast_math() -> None:
    """TF32 matmuls/convs (T2); bf16 autocast is applied around the forward passes."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def gpu_mem_gb(device) -> float:
    import torch
    if device.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 1e9


# ---------------------------------------------------------------- files
def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_parquet(path) -> list[Path]:
    """A parquet file, or every *.parquet under a directory (sorted, recursive)."""
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(p.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {p}")
    return files


def find_file(root, name: str) -> Path | None:
    p = Path(root)
    if (p / name).exists():
        return p / name
    hits = sorted(p.rglob(name))
    return hits[0] if hits else None


def resolve_model_artifact(path) -> Path:
    """A training job's output (`model.tar.gz`, possibly inside a channel dir) or an already extracted directory."""
    p = Path(path)
    tar = p if p.is_file() else find_file(p, "model.tar.gz")
    if tar is None:
        return p
    dst = Path(os.environ.get("TMPDIR", "/tmp")) / "ber_model_extracted"
    # torchrun starts one process per GPU: only rank 0 extracts (into a temp dir, renamed when complete), the others
    # wait at the barrier and then find the finished folder. Concurrent extraction into one folder raced (2026-09-27).
    if is_main():
        if not (dst / "run_config.json").exists() and not any(dst.rglob("run_config.json")):
            tmp = dst.with_name(dst.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(dst, ignore_errors=True)
            tmp.mkdir(parents=True)
            with tarfile.open(tar) as t:
                t.extractall(tmp)
            os.replace(tmp, dst)
            log(f"extracted {tar} -> {dst}")
    barrier()
    hit = find_file(dst, "run_config.json")
    if hit is None:
        raise FileNotFoundError(f"run_config.json not found under {dst} after extracting {tar}")
    return hit.parent


def default_checkpoint_dir(local: str) -> str:
    return SM_CHECKPOINT_DIR if ON_SAGEMAKER else local
