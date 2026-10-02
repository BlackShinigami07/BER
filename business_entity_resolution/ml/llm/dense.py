"""Dense multilingual retrieval for India (DENSE_RECALL_PLAN.md), run as a SageMaker job under torchrun (one process per GPU).

Inputs (channel dir, --in-dir): for each split S in --splits: q_S.parquet (id, text) = S1 queries, p_S.parquet (id, text)
= pool records. Every rank embeds its 1/world share of the pool and of the queries with a multilingual E5 model (mean
pooling, L2-normalised, fp16), writes them to local disk, waits for the others, loads the whole pool matrix on its GPU
and searches its query share exactly (matmul + top-k). Output: --final-dir/dense_hits_S_r<rank>.parquet with
(s1_id, cand_id, rank, cos) for hits with cos >= --min-cos, top --k per query.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [rank {os.environ.get('RANK', '0')}] {msg}", flush=True)


def wait_for(paths: list[Path], what: str, timeout_s: int = 7200) -> None:
    t0 = time.time()
    while not all(p.exists() for p in paths):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"waiting for {what}")
        time.sleep(2)


def embed(model, tok, texts: list[str], prefix: str, bs: int, max_len: int, dev) -> torch.Tensor:
    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16)
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: little padding
    t0 = time.time()
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([prefix + texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        v = torch.nn.functional.normalize(v.float(), dim=-1).half()
        out[torch.as_tensor(idx)] = v.cpu()
        if (i // bs) % 200 == 0:
            done = min(i + bs, len(texts))
            log(f"  embedded {done:,}/{len(texts):,} ({done / max(1e-9, time.time() - t0):,.0f}/s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/opt/ml/input/data/dense_in")
    ap.add_argument("--work-dir", default="/tmp/dense")
    ap.add_argument("--final-dir", default="/opt/ml/model")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--model-id", default="intfloat/multilingual-e5-base")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min-cos", type=float, default=0.80)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--q-chunk", type=int, default=1024)
    a, _ = ap.parse_known_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{local}")
    torch.cuda.set_device(dev)
    work, final = Path(a.work_dir), Path(a.final_dir)
    work.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    flag = work / "model.ready"
    if rank == 0:
        path = snapshot_download(a.model_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
        flag.write_text(path)
    wait_for([flag], "model download")
    path = flag.read_text()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(dev).eval()
    log(f"model {a.model_id} on {dev}; world {world}")
    for split in [s for s in a.splits.split(",") if s]:
        t0 = time.time()
        q = pq.read_table(Path(a.in_dir) / f"q_{split}.parquet").to_pydict()
        p = pq.read_table(Path(a.in_dir) / f"p_{split}.parquet").to_pydict()
        nq, npool = len(q["id"]), len(p["id"])
        log(f"[{split}] {nq:,} queries, {npool:,} pool records")
        p_lo, p_hi = rank * npool // world, (rank + 1) * npool // world
        q_lo, q_hi = rank * nq // world, (rank + 1) * nq // world
        pe = embed(model, tok, p["text"][p_lo:p_hi], "passage: ", a.batch, a.max_len, dev)
        torch.save(pe, work / f"p_{split}_{rank}.pt")
        (work / f"p_{split}_{rank}.done").write_text("1")
        qe = embed(model, tok, q["text"][q_lo:q_hi], "query: ", a.batch, a.max_len, dev)
        log(f"[{split}] embedded in {time.time() - t0:.0f}s; waiting for the other ranks")
        wait_for([work / f"p_{split}_{r}.done" for r in range(world)], "pool embeddings")
        P = torch.cat([torch.load(work / f"p_{split}_{r}.pt") for r in range(world)]).to(dev)
        s1s, cands, ranks, coss = [], [], [], []
        for i in range(0, qe.shape[0], a.q_chunk):
            sim = qe[i:i + a.q_chunk].to(dev) @ P.T
            val, ix = torch.topk(sim, a.k, dim=1)
            val, ix = val.float().cpu().numpy(), ix.cpu().numpy()
            for r in range(val.shape[0]):
                keep = val[r] >= a.min_cos
                n = int(keep.sum())
                if n:
                    s1s.extend([q["id"][q_lo + i + r]] * n)
                    cands.extend(p["id"][j] for j in ix[r][keep])
                    ranks.extend(range(1, n + 1))
                    coss.extend(val[r][keep].tolist())
            del sim
        del P
        torch.cuda.empty_cache()
        t = pa.table({"s1_id": s1s, "cand_id": cands, "rank": pa.array(ranks, pa.int32()), "cos": pa.array(coss, pa.float32())})
        pq.write_table(t, final / f"dense_hits_{split}_r{rank}.parquet")
        log(f"[{split}] {t.num_rows:,} hits for {q_hi - q_lo:,} queries in {time.time() - t0:.0f}s")
    (work / f"all_{rank}.done").write_text("1")
    wait_for([work / f"all_{r}.done" for r in range(world)], "all ranks")
    log("done")


if __name__ == "__main__":
    main()
