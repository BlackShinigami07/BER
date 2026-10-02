"""WP-B: build pre-tokenised stage-B blocks (GEMMA_PLAN.md section 2) for train / dev / valid / test.

    uv run python -m business_entity_resolution.ml blocks --artifacts-dir artifacts --n-train 400000
    uv run python -m business_entity_resolution.ml blocks --artifacts-dir dev/mini_artifacts --n-train 2000 --n-dev 300 --n-valid 1000

One block per S1 entity: the S1 line then its top-K candidates by stage-A probability (K=12, plus p >= 0.05 up to 16),
each candidate line ending in the " ans" marker whose hidden state the answer head reads. Output per subset:
artifacts/{train,test}/ml/blocks/{subset}/blocks_{subset}_NNNN.parquet + blocks_meta.json, and ml/blocks_stats.json.

Dataset-size abstraction: the train sample is drawn once with the quotas of section 2.4 at the largest size you may
want (`--n-train`), then randomly permuted into `sample_rank`. Any prefix sample_rank < N is itself a quota-respecting
sample, and shard files are cut in sample_rank order, so the trainer's `--n-blocks N` reads only the files it needs.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ..io import load_frame, save_json
from ..progress import fmt_secs, log, pbar, stage
from .common import CAND_COLS, SUBSET_SPLIT, attach_p_gbm, blocks_dir, candidate_part_files, load_folds, ml_dir, subset_split

TEMPLATE_VERSION = 1
MARKER = " ans"


@dataclass
class Quotas:
    india: float = 0.55       # country share of the train/dev sample (rest: US)
    indic: float = 0.30       # >= share of blocks with an Indic-script candidate (all from India)
    hard: float = 0.50        # >= share with max p in [0.05, 0.95] or a wrong stage-A decision
    empty: float = 0.10       # >= share whose top candidate (or a true match) has an empty address
    singleton: float = 0.08   # exact share of entities with no true match


@dataclass
class BlocksConfig:
    artifacts_dir: Path = Path("artifacts")
    subsets: tuple[str, ...] = ("train", "dev", "valid", "test")
    n_train: int = 250_000
    n_dev: int = 5_000
    n_valid: int = 30_000
    tokenizer: str = "google/gemma-4-E4B"
    k: int = 12
    k_max: int = 16
    p_extra: float = 0.05
    max_tokens: int = 800
    addr_chars: int = 120
    name_chars: int = 120
    shard_blocks: int = 50_000
    tok_chunk: int = 5_000
    seed: int = 42
    p_source: str = "proxy"   # p in the prompts: the proxy the stage-B model was trained with (CATBOOST_PLAN.md intro)
    quotas: Quotas = field(default_factory=Quotas)


# ---------------------------------------------------------------- entity table + candidate selection
def _texts(split_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    clean = lambda c: pl.col(c).fill_null("").str.replace_all(r"[\r\n\t]+", " ").str.strip_chars()
    s1 = load_frame(split_dir / "normalized_source1.pkl").select(
        pl.col("entity_id").alias("s1_id"), clean("business_name").alias("name"), clean("business_address").alias("addr"),
        clean("name_core").alias("core"), pl.col("name_script").fill_null("latin").alias("script"))
    pool = pl.concat([load_frame(split_dir / f"normalized_source{i}.pkl").select(
        pl.col("entity_id").alias("cand_id"), clean("business_name").alias("name"), clean("business_address").alias("addr"),
        clean("name_core").alias("core"), pl.col("name_script").fill_null("latin").alias("script")) for i in (2, 3)])
    return s1, pool


def entity_tables(cfg: BlocksConfig, split: str, pool: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """Per-entity features (sampling quotas, slices, routing flags) and the selected candidate rows."""
    split_dir = cfg.artifacts_dir / split
    labelled = split == "train"
    folds = load_folds(cfg.artifacts_dir) if labelled else None
    meta = pool.select("cand_id", pl.col("script").alias("c_script"), (pl.col("addr") == "").alias("addr_empty")).lazy()
    ents, sels, source = [], [], "?"
    files = candidate_part_files(split_dir)
    for f in pbar(files, desc=f"select {split}", unit="part"):
        cols = CAND_COLS + (["label"] if labelled else [])
        lf, source = attach_p_gbm(pl.scan_parquet(f).select(cols), cfg.artifacts_dir, split, folds, source=cfg.p_source)
        df = (lf.join(meta, on="cand_id", how="left").collect()
                .sort(["s1_id", "p_gbm", "rrf_rank"], descending=[False, True, False])
                .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("sel_rank")))
        sel = df.filter((pl.col("sel_rank") <= cfg.k) | ((pl.col("sel_rank") <= cfg.k_max) & (pl.col("p_gbm") >= cfg.p_extra)))
        agg = [pl.col("country").first(), pl.len().alias("n_sel"), pl.col("p_gbm").max().alias("max_p"),
               (pl.col("p_gbm") > 0.3).sum().alias("n_over03"), (pl.col("c_script") == "indic").any().alias("has_indic"),
               pl.col("addr_empty").fill_null(False).first().alias("top_empty"),
               pl.col("cos_char_name").fill_null(0.0).max().alias("max_cos_name")]
        if labelled:
            agg += [pl.col("fold").first(), pl.col("label").cast(pl.Int32).sum().alias("n_pos_sel"),
                    ((pl.col("p_gbm") >= 0.5).cast(pl.Int8) != pl.col("label").cast(pl.Int8)).any().alias("gbm_wrong"),
                    ((pl.col("label") == 1) & pl.col("addr_empty").fill_null(False)).any().alias("pos_empty")]
        e = sel.group_by("s1_id", maintain_order=True).agg(agg)
        if labelled:
            e = e.join(df.group_by("s1_id").agg(pl.col("label").cast(pl.Int32).sum().alias("n_pos_all")), on="s1_id", how="left")
        ents.append(e)
        keep = ["s1_id", "cand_id", "src", "sel_rank", "p_gbm", "n_blockers"] + (["label"] if labelled else [])
        sels.append(sel.select(keep))
    ent, sel = pl.concat(ents), pl.concat(sels)
    band = pl.col("max_p").is_between(0.05, 0.95)
    route = dict(route_indic=(pl.col("country") == "India") & pl.col("has_indic"),
                 route_uncertain=band | (pl.col("n_over03") >= 2),
                 route_weak=pl.col("top_empty") | (pl.col("max_cos_name") < 0.3),
                 route_france=pl.col("country") == "France")
    ent = ent.with_columns(**route).with_columns(pl.any_horizontal(list(route)).alias("routed"))
    if labelled:
        missed_f = split_dir / "missed_pairs.pkl"
        missed = (load_frame(missed_f).group_by("s1_id").agg(pl.len().cast(pl.Int32).alias("n_missed"))
                  if missed_f.exists() else pl.DataFrame(schema={"s1_id": pl.String, "n_missed": pl.Int32}))
        ent = (ent.join(missed, on="s1_id", how="left")
                  .with_columns((pl.col("n_pos_all") + pl.col("n_missed").fill_null(0)).alias("n_truth"))
                  .with_columns((pl.col("n_truth") == 0).alias("singleton"),
                                # label-aware hardness only for ml_train entities (sampling); label-free elsewhere
                                pl.when(pl.col("fold") == "ml_train").then(band | pl.col("gbm_wrong")).otherwise(band).alias("is_hard"),
                                # empty-address quota: the plan's "top candidate empty" is almost only singletons (~0.5% of
                                # non-singleton entities), so sampling also counts blocks where a true match has no address
                                pl.when(pl.col("fold") == "ml_train").then(pl.col("top_empty") | pl.col("pos_empty"))
                                  .otherwise(pl.col("top_empty")).alias("empty_addr")))
    else:
        ent = ent.with_columns(band.alias("is_hard"), pl.col("top_empty").alias("empty_addr"))
    log(f"[{split}] {ent.height:,} entities, {sel.height:,} selected candidates (p_gbm source: {source})")
    return ent, sel, source


# ---------------------------------------------------------------- sampling (section 2.4)
def quota_sample(ent: pl.DataFrame, avail: np.ndarray, n: int, q: Quotas, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a sample meeting the country split, the >= quotas (Indic, hard, empty address) and the exact
    singleton share, drawn without replacement from rows where `avail`; remainder random."""
    country = ent["country"].to_numpy()
    single = ent["singleton"].to_numpy()
    flags = {"indic": ent["has_indic"].to_numpy(), "hard": ent["is_hard"].to_numpy(), "empty": ent["empty_addr"].to_numpy()}
    shares = {c: s for c, s in (("India", q.india), ("US", 1 - q.india)) if (avail & (country == c)).any()}
    if not shares:
        return np.empty(0, dtype=np.int64)
    n = min(n, int(avail.sum()))
    tot = sum(shares.values())
    targets = {c: int(round(n * s / tot)) for c, s in shares.items()}
    targets[list(targets)[-1]] += n - sum(targets.values())
    pool = avail.copy()
    chosen: list[np.ndarray] = []
    for c, n_c in targets.items():
        in_c = pool & (country == c)
        n_c = min(n_c, int(in_c.sum()))
        taken: list[np.ndarray] = []

        def draw(mask: np.ndarray, k: int) -> None:
            idx = np.nonzero(pool & (country == c) & mask)[0]
            k = min(max(0, int(k)), len(idx), n_c - sum(len(t) for t in taken))
            if k > 0:
                t = rng.choice(idx, size=k, replace=False)
                pool[t] = False
                taken.append(t)

        draw(single, round(q.singleton * n_c))
        want = {"indic": q.indic * n if c == "India" else 0.0, "hard": q.hard * n_c, "empty": q.empty * n_c}
        for name, tgt in want.items():
            have = sum(int(flags[name][t].sum()) for t in taken)
            draw(flags[name] & ~single, math.ceil(tgt) - have)
            got = sum(int(flags[name][t].sum()) for t in taken)
            if got < math.ceil(tgt):
                log(f"WARNING: quota '{name}' for {c}: {got:,} of {math.ceil(tgt):,} wanted (pool exhausted)")
        draw(~single, n_c)                       # remainder: random non-singletons
        draw(np.ones_like(single), n_c)          # only if non-singletons ran out
        chosen.append(np.concatenate(taken) if taken else np.empty(0, dtype=np.int64))
    return np.concatenate(chosen)


def stratified_sample(ent: pl.DataFrame, avail: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    country = ent["country"].to_numpy()
    idx = np.nonzero(avail)[0]
    if n >= len(idx):
        return idx
    out = []
    for c in np.unique(country[idx]):
        ic = idx[country[idx] == c]
        out.append(rng.choice(ic, size=int(round(n * len(ic) / len(idx))), replace=False))
    return np.concatenate(out)


# ---------------------------------------------------------------- text + tokenisation (sections 2.2, 2.3)
def _p2(col: str) -> pl.Expr:
    pint = (pl.col(col).clip(0.0, 1.0) * 100).round(0).cast(pl.Int32)
    return pl.format("{}.{}", pint // 100, (pint % 100).cast(pl.String).str.zfill(2))


def _name_hint(cfg: BlocksConfig) -> pl.Expr:
    name = pl.col("name").str.slice(0, cfg.name_chars)
    hint = (pl.when((pl.col("script") != "latin") & (pl.col("core") != ""))
              .then(pl.format(" ({})", pl.col("core").str.slice(0, cfg.name_chars))).otherwise(pl.lit("")))
    return pl.concat_str([name, hint])


def _addr(cfg: BlocksConfig) -> pl.Expr:
    a = pl.col("addr").str.slice(0, cfg.addr_chars)
    return pl.when(a == "").then(pl.lit("-")).otherwise(a)


class Tokenizer:
    """The Gemma tokenizer + the ids the contract needs (BOS, pad, marker, line prefixes)."""

    def __init__(self, name: str, k_max: int):
        from transformers import AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.bos = int(self.tok.bos_token_id)
        self.pad = int(self.tok.pad_token_id if self.tok.pad_token_id is not None else 0)
        self.prefix = [self.encode([f"\n[{j}]"])[0] for j in range(1, k_max + 1)]
        probe = self.encode([" x | - | S2 | p=0.12 nb=3" + MARKER])[0]
        self.marker = int(probe[-1])
        if not self.tok.convert_ids_to_tokens(self.marker).lstrip("▁ ").endswith("ans"):
            raise RuntimeError(f"marker token {self.marker} does not decode to 'ans'")

    def encode(self, texts: list[str]) -> list[list[int]]:
        return self.tok(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]

    def encode_flat(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ids = self.encode(texts)
        lens = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        off = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        flat = np.fromiter((t for x in ids for t in x), dtype=np.int32, count=int(off[-1]))
        return flat, off


def build_shard(cfg: BlocksConfig, tk: Tokenizer, ents: pl.DataFrame, sel: pl.DataFrame, s1_text: pl.DataFrame,
                pool: pl.DataFrame, labelled: bool, stats: dict):
    """Tokenise the blocks of `ents` (in that order) and return them as a pyarrow table sorted by n_tokens."""
    import pyarrow as pa
    order = ents.select("s1_id").with_row_index("_e")
    s1 = (order.join(s1_text, on="s1_id", how="left").join(ents.select("s1_id", "country"), on="s1_id", how="left").sort("_e")
               .with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null(""), pl.col("core").fill_null(""),
                             pl.col("script").fill_null("latin"))
               .with_columns(pl.format("S1: {} | {} | {}", pl.col("country"), _name_hint(cfg), _addr(cfg)).alias("line")))
    rows = (sel.join(order, on="s1_id", how="inner").join(pool, on="cand_id", how="left")
               .with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null(""), pl.col("core").fill_null(""),
                             pl.col("script").fill_null("latin"))
               .sort(["_e", "sel_rank"])
               .with_columns(pl.format(" {} | {} | {} | p={} nb={}" + MARKER, _name_hint(cfg), _addr(cfg), pl.col("src"),
                                       _p2("p_gbm"), pl.col("n_blockers")).alias("line")))
    n_e = ents.height
    l_off = np.zeros(n_e + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows["_e"].to_numpy(), minlength=n_e), out=l_off[1:])
    if np.any(np.diff(l_off) == 0):
        raise RuntimeError("an entity without candidates reached block building")
    cand = rows["cand_id"].to_numpy()
    p = rows["p_gbm"].to_numpy().astype(np.float32)
    lab = rows["label"].to_numpy().astype(np.int8) if labelled else None
    s1_lines, c_lines = s1["line"].to_list(), rows["line"].to_list()
    plen = np.array([len(x) for x in tk.prefix], dtype=np.int64)
    prefix = [np.asarray(x, dtype=np.int32) for x in tk.prefix]
    bos = np.array([tk.bos], dtype=np.int32)
    out_ids, out_mpos, out_keep, n_tok, s1_len = [], [], np.zeros(n_e, dtype=np.int64), np.zeros(n_e, dtype=np.int64), np.zeros(n_e, dtype=np.int64)
    for a in range(0, n_e, cfg.tok_chunk):
        b = min(n_e, a + cfg.tok_chunk)
        sf, so = tk.encode_flat(s1_lines[a:b])
        cf, co = tk.encode_flat(c_lines[l_off[a]:l_off[b]])
        for e in range(a, b):
            s = sf[so[e - a]:so[e - a + 1]]
            la, lb = l_off[e] - l_off[a], l_off[e + 1] - l_off[a]
            n = lb - la
            lens = plen[:n] + (co[la + 1:lb + 1] - co[la:lb])
            head = 1 + len(s)
            tot = head + np.cumsum(lens)
            keep = max(1, int(np.searchsorted(tot, cfg.max_tokens, side="right")))
            pieces = [bos, s]
            for j in range(keep):
                pieces += [prefix[j], cf[co[la + j]:co[la + j + 1]]]
            ids = np.concatenate(pieces)
            out_ids.append(ids)
            out_mpos.append((tot[:keep] - 1).astype(np.int16))
            out_keep[e], n_tok[e], s1_len[e] = keep, len(ids), head
            if keep < n:
                stats["capped_blocks"] += 1
                stats["capped_cands"] += int(n - keep)
                if labelled:
                    stats["capped_pos"] += int(lab[l_off[e] + keep:l_off[e + 1]].sum())
    # candidate-level arrays restricted to the kept lines
    kept_mask = np.zeros(len(cand), dtype=bool)
    for e in range(n_e):
        kept_mask[l_off[e]:l_off[e] + out_keep[e]] = True
    k_off = np.zeros(n_e + 1, dtype=np.int64)
    np.cumsum(out_keep, out=k_off[1:])
    ids_flat = np.concatenate(out_ids)
    mpos_flat = np.concatenate(out_mpos)
    i_off = np.zeros(n_e + 1, dtype=np.int64)
    np.cumsum(n_tok, out=i_off[1:])
    if np.any(ids_flat[np.repeat(i_off[:-1], out_keep) + mpos_flat] != tk.marker):
        raise RuntimeError("marker positions do not point at the marker token")
    lst = lambda off, values: pa.ListArray.from_arrays(pa.array(off.astype(np.int32)), values)
    cols = {"s1_id": pa.array(ents["s1_id"].to_list(), pa.string()),
            "input_ids": lst(i_off, pa.array(ids_flat, pa.int32())),
            "marker_pos": lst(k_off, pa.array(mpos_flat, pa.int16())),
            "s1_len": pa.array(s1_len.astype(np.int16)),
            "cand_ids": lst(k_off, pa.array(cand[kept_mask].tolist(), pa.string())),
            "p_gbm": lst(k_off, pa.array(p[kept_mask], pa.float32())),
            "n_tokens": pa.array(n_tok.astype(np.int16)),
            "n_cands": pa.array(out_keep.astype(np.int8))}
    if labelled:
        kl = lab[kept_mask]
        cols["labels"] = lst(k_off, pa.array(kl, pa.int8()))
        cols["n_pos_block"] = pa.array(np.add.reduceat(kl.astype(np.int64), k_off[:-1]).astype(np.int16))
        cols["n_truth"] = pa.array(ents["n_truth"].to_numpy().astype(np.int16))
        cols["singleton"] = pa.array(ents["singleton"].to_numpy())
    for c in ("country", "has_indic", "is_hard", "top_empty", "empty_addr", "routed", "route_indic", "route_uncertain", "route_weak",
              "route_france", "max_p", "sample_rank"):
        if c in ents.columns:
            v = ents[c].to_numpy()
            cols[c] = pa.array(v.astype(np.float32) if c == "max_p" else (v.astype(np.int32) if c == "sample_rank" else v))
    t = pa.table(cols)
    return t.take(np.argsort(n_tok, kind="stable")), n_tok


def write_subset(cfg: BlocksConfig, tk: Tokenizer, subset: str, ents: pl.DataFrame, sel: pl.DataFrame, s1_text: pl.DataFrame,
                 pool: pl.DataFrame, p_source: str) -> dict:
    import pyarrow.parquet as pq
    out = blocks_dir(cfg.artifacts_dir, subset)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"blocks_{subset}_*.parquet"):
        old.unlink()
    labelled = subset_split(subset) == "train"
    sel = sel.join(ents.select("s1_id"), on="s1_id", how="semi")
    pool = pool.join(sel.select("cand_id").unique(), on="cand_id", how="semi")
    stats = {"capped_blocks": 0, "capped_cands": 0, "capped_pos": 0}
    files, all_tok = [], []
    n_shards = max(1, math.ceil(ents.height / cfg.shard_blocks))
    for i in pbar(range(n_shards), desc=f"tokenise {subset}", unit="shard"):
        chunk = ents.slice(i * cfg.shard_blocks, cfg.shard_blocks)
        t, n_tok = build_shard(cfg, tk, chunk, sel, s1_text, pool, labelled, stats)
        f = out / f"blocks_{subset}_{i:04d}.parquet"
        pq.write_table(t, f, row_group_size=len(t))
        info = {"file": f.name, "rows": t.num_rows}
        if "sample_rank" in chunk.columns:
            info.update(min_rank=int(chunk["sample_rank"].min()), max_rank=int(chunk["sample_rank"].max()))
        if "routed" in chunk.columns:
            info["n_routed"] = int(chunk["routed"].sum())
        files.append(info)
        all_tok.append(n_tok)
    n_tok = np.concatenate(all_tok)
    meta = {"tokenizer": tk.name, "template_version": TEMPLATE_VERSION, "pad_id": tk.pad, "bos_id": tk.bos, "marker_id": tk.marker,
            "prefix_ids": tk.prefix, "max_tokens": cfg.max_tokens, "k": cfg.k, "k_max": cfg.k_max, "p_extra": cfg.p_extra,
            "addr_chars": cfg.addr_chars, "name_chars": cfg.name_chars, "subset": subset, "split": subset_split(subset),
            "p_source": p_source, "n_blocks": int(ents.height), "files": files, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(meta, out / "blocks_meta.json")
    st = {"n_blocks": int(ents.height), "tokens_p50": float(np.median(n_tok)), "tokens_p90": float(np.percentile(n_tok, 90)),
          "tokens_p95": float(np.percentile(n_tok, 95)), "tokens_p99": float(np.percentile(n_tok, 99)), "tokens_max": int(n_tok.max()),
          "tokens_total": int(n_tok.sum()), **stats, "files": len(files)}
    st.update(_shares(ents))
    if "sample_rank" in ents.columns and subset == "train":
        srt = ents.sort("sample_rank")
        st["prefix_shares"] = {str(n): _shares(srt.head(n)) for n in (1_000, 10_000, 50_000, 100_000, 150_000, 250_000) if n < ents.height}
    if labelled:
        cov = sel.group_by("s1_id").agg(pl.col("label").cast(pl.Int32).sum().alias("in_sel")).join(
            ents.select("s1_id", "n_pos_all", "n_truth"), on="s1_id")
        st["label_rate_in_blocks"] = float(sel["label"].cast(pl.Float64).mean())
        st["positives_in_blocks_over_retrieved"] = float(cov["in_sel"].sum() / max(1, cov["n_pos_all"].sum()))
        st["positives_in_blocks_over_truth"] = float(cov["in_sel"].sum() / max(1, cov["n_truth"].sum()))
    log(f"[{subset}] {ents.height:,} blocks in {len(files)} file(s) -> {out} | tokens p50 {st['tokens_p50']:.0f} p95 {st['tokens_p95']:.0f} "
        f"max {st['tokens_max']} | capped blocks {stats['capped_blocks']:,}")
    return st


def _shares(e: pl.DataFrame) -> dict:
    n = max(1, e.height)
    out = {"n": e.height, "country": {c: round(k / n, 4) for c, k in e["country"].value_counts().iter_rows()}}
    for c in ("has_indic", "is_hard", "top_empty", "empty_addr", "singleton", "routed"):
        if c in e.columns:
            out[c] = round(float(e[c].sum()) / n, 4)
    return out


# ---------------------------------------------------------------- driver
def run(cfg: BlocksConfig) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    tk = Tokenizer(cfg.tokenizer, cfg.k_max)
    log(f"tokenizer {cfg.tokenizer}: bos {tk.bos} pad {tk.pad} marker {tk.marker} ({tk.tok.convert_ids_to_tokens(tk.marker)!r})")
    report = {"config": {**asdict(cfg), "artifacts_dir": str(cfg.artifacts_dir)}}
    for split in ("train", "test"):
        subsets = [s for s in cfg.subsets if SUBSET_SPLIT[s] == split]
        if not subsets:
            continue
        with stage(f"blocks {split}: {','.join(subsets)}"):
            s1_text, pool = _texts(cfg.artifacts_dir / split)
            ent, sel, source = entity_tables(cfg, split, pool)
            picks: dict[str, pl.DataFrame] = {}
            if split == "train":
                fold = ent["fold"].to_numpy()
                avail = fold == "ml_train"
                idx = quota_sample(ent, avail, cfg.n_train, cfg.quotas, rng)
                idx = idx[rng.permutation(len(idx))]
                picks["train"] = ent[idx].with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("sample_rank"))
                avail[idx] = False
                didx = quota_sample(ent, avail, cfg.n_dev, cfg.quotas, rng)
                picks["dev"] = ent[didx[rng.permutation(len(didx))]].with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("sample_rank"))
                vidx = stratified_sample(ent, fold == "ml_valid", cfg.n_valid, rng)
                picks["valid"] = ent[np.sort(vidx)]
                if len(idx) < cfg.n_train:
                    log(f"WARNING: only {len(idx):,} ml_train entities available for n_train={cfg.n_train:,}")
            else:
                picks["test"] = ent.sort(["routed", "s1_id"], descending=[True, False])
            for s in subsets:
                report[s] = write_subset(cfg, tk, s, picks[s], sel, s1_text, pool, source)
            save_json({k: v for k, v in report.items() if k == "config" or SUBSET_SPLIT.get(k) == split},
                      ml_dir(cfg.artifacts_dir, split) / "blocks_stats.json")
    log(f"blocks done in {fmt_secs(time.time() - t0)}")
    return report


def build_parser() -> argparse.ArgumentParser:
    d = BlocksConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", default=str(d.artifacts_dir))
    p.add_argument("--subsets", default=",".join(d.subsets), help="any of train,dev,valid,test")
    p.add_argument("--n-train", type=int, default=d.n_train, help="size of the train sample (build the largest you may use)")
    p.add_argument("--n-dev", type=int, default=d.n_dev)
    p.add_argument("--n-valid", type=int, default=d.n_valid)
    p.add_argument("--tokenizer", default=d.tokenizer)
    p.add_argument("--k", type=int, default=d.k)
    p.add_argument("--k-max", type=int, default=d.k_max)
    p.add_argument("--p-extra", type=float, default=d.p_extra)
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--addr-chars", type=int, default=d.addr_chars)
    p.add_argument("--name-chars", type=int, default=d.name_chars)
    p.add_argument("--shard-blocks", type=int, default=d.shard_blocks)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--p-source", default=d.p_source, choices=["proxy", "gbm", "auto"],
                   help="stage-A p inside the prompts; gbm needs out-of-fold CatBoost p for ml_train (gbm_oof_train.parquet)")
    for f in Quotas.__dataclass_fields__:
        p.add_argument(f"--quota-{f}", type=float, default=getattr(Quotas(), f))
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    cfg = BlocksConfig(artifacts_dir=Path(a.artifacts_dir), subsets=tuple(s for s in a.subsets.split(",") if s),
                       n_train=a.n_train, n_dev=a.n_dev, n_valid=a.n_valid, tokenizer=a.tokenizer, k=a.k, k_max=a.k_max,
                       p_extra=a.p_extra, max_tokens=a.max_tokens, addr_chars=a.addr_chars, name_chars=a.name_chars,
                       shard_blocks=a.shard_blocks, seed=a.seed, p_source=a.p_source,
                       quotas=Quotas(**{f: getattr(a, f"quota_{f}") for f in Quotas.__dataclass_fields__}))
    run(cfg)


if __name__ == "__main__":
    main()
