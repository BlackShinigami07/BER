"""Stage `block`: memory-lean candidate generation.

Per country -> per state shard (plan D6): a shard's queries are scored against the pool rows of the same state plus
the pool rows whose state is unknown; queries without a state are scored against the whole country pool. For every
shard the four blockers run (B0 exact, B1 word TF-IDF on CPU, B2/B3 char 3-gram TF-IDF on GPU when available,
optional B4 dense), their hits are fused with weighted RRF, capped, labelled (train) and written as one parquet part.
TF-IDF vectorizers are fitted once per country; transforms run per shard in a process pool that is created once per
blocker and reused. Nothing larger than one shard's hits is ever materialised."""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from ..config import BLOCKER_NAMES, BlockingConfig
from ..io import (candidate_parts, candidate_parts_dir, gt_to_pairs, list_countries, load_pkl, normalized_part,
                  read_ground_truth, safe_name, save_json, save_pkl)
from ..normalize import AddressNormalizer, string_to_components
from ..prepare import _run_chunks
from ..progress import fmt_secs, log, pbar, stage
from .exact import exact_blocker
from .tfidf import _init_transform, _transform_chunk, cpu_topk, fit_vectorizer, hits_frame
from .union import rrf_union

CAND_COLS = ["s1_id", "cand_id", "src", "country", "state_match", "n_blockers", "rrf_score", "rrf_rank"] + \
            [f"rank_{b}" for b in BLOCKER_NAMES] + [f"cos_{b}" for b in BLOCKER_NAMES]


# ------------------------------------------------------------------ GPU decision
def use_gpu(cfg: BlockingConfig) -> bool:
    if cfg.gpu is False:
        return False
    try:
        from .gpu_topk import gpu_available, gpu_name
        ok = gpu_available()
    except Exception as e:  # torch not installed
        ok = False
        if cfg.gpu is True:
            log(f"WARNING: --gpu requested but torch/CUDA unavailable ({e}); falling back to CPU")
        return False
    if ok:
        log(f"GPU enabled for char 3-gram passes: {gpu_name()}")
    elif cfg.gpu is True:
        log("WARNING: --gpu requested but no CUDA device found; falling back to CPU")
    return ok


# ------------------------------------------------------------------ loading + sampling
def load_country(split_dir, country: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["entity_id", "name_core", "addr_comps"]
    q = pl.read_parquet(normalized_part(split_dir, "source1", country), columns=cols)
    p = pl.concat([pl.read_parquet(normalized_part(split_dir, s, country), columns=cols) for s in ("source2", "source3")
                   if normalized_part(split_dir, s, country).exists()])
    return q, p


def subsample_country(q: pl.DataFrame, p: pl.DataFrame, pairs: pl.DataFrame | None, cfg: BlockingConfig, n_countries: int):
    per = max(1, cfg.sample_s1 // n_countries)
    q = q.sample(n=min(per, q.height), seed=cfg.seed)
    keep: list[str] = []
    if pairs is not None:
        pairs = pairs.join(q.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
        keep = pairs["cand_id"].to_list()
    tm = p.filter(pl.col("entity_id").is_in(keep)) if keep else p.head(0)
    rnd = p.filter(~pl.col("entity_id").is_in(keep)).sample(fraction=cfg.pool_fraction, seed=cfg.seed) if keep else p.sample(fraction=cfg.pool_fraction, seed=cfg.seed)
    p = pl.concat([tm, rnd])
    log(f"sample mode: {q.height:,} S1, pool {p.height:,} ({tm.height:,} true matches + {rnd.height:,} random @ {cfg.pool_fraction})")
    return q, p, pairs


# ------------------------------------------------------------------ address finalisation (process pool)
_ADDR_NORM: AddressNormalizer | None = None


def _init_addr_worker(alias: dict[str, str]):
    global _ADDR_NORM
    _ADDR_NORM = AddressNormalizer(alias=alias)


def _addr_chunk(comps_list: list[str]):
    an = _ADDR_NORM or AddressNormalizer()
    norms, states = [], []
    for comps in comps_list:
        a, s = an(string_to_components(comps))
        norms.append(a); states.append(s)
    return norms, states


def finalize_addresses(df: pl.DataFrame, alias: dict[str, str], tag: str, cfg: BlockingConfig) -> pl.DataFrame:
    payloads = [df["addr_comps"].slice(i, cfg.norm_chunk_size).to_list() for i in range(0, df.height, cfg.norm_chunk_size)]
    parts = _run_chunks(_addr_chunk, payloads, cfg.n_workers(), f"addresses[{tag}]", _init_addr_worker, (alias,))
    norms = [x for p in parts for x in p[0]]; states = [x for p in parts for x in p[1]]
    return df.with_columns(pl.Series("addr_norm", norms), pl.Series("state_norm", states, dtype=pl.Utf8)).drop("addr_comps")


# ------------------------------------------------------------------ shard planning
def plan_groups(q_state: list[str | None], p_state: list[str | None], shard: bool):
    """[(name, q_sel, p_sel)] with global row indices. Unknown-state pool rows join every shard; unknown-state
    queries search the whole pool."""
    qs = np.asarray([s or "" for s in q_state]); ps = np.asarray([s or "" for s in p_state])
    if not shard:
        return [("all", np.arange(len(qs)), np.arange(len(ps)))]
    p_unknown = np.flatnonzero(ps == "")
    groups = []
    for s in sorted(set(qs[qs != ""].tolist())):
        q_sel = np.flatnonzero(qs == s)
        p_sel = np.sort(np.concatenate([np.flatnonzero(ps == s), p_unknown]))
        if p_sel.size:
            groups.append((safe_name(s), q_sel, p_sel))
    q_unknown = np.flatnonzero(qs == "")
    if q_unknown.size:
        groups.append(("unknown", q_unknown, np.arange(len(ps))))
    return groups


# ------------------------------------------------------------------ transform helper with a persistent pool
class Transformer:
    """Holds one fitted vectorizer and (optionally) a process pool initialised with it."""

    def __init__(self, vec, workers: int, chunk: int):
        self.vec = vec; self.chunk = chunk
        # each Transformer owns a pool whose workers hold *this* vectorizer; inline transforms use self.vec directly
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_transform, initargs=(vec,)) if workers > 1 else None

    def __call__(self, texts: np.ndarray, tag: str):
        import scipy.sparse as sp
        n = len(texts)
        if n == 0:
            return sp.csr_matrix((0, len(self.vec.vocabulary_)), dtype=np.float32)
        chunks = [texts[i:i + self.chunk].tolist() for i in range(0, n, self.chunk)]
        if self.pool is None or len(chunks) == 1:
            mats = [self.vec.transform(c).tocsr() for c in chunks]
        else:
            mats = list(self.pool.map(_transform_chunk, chunks))
        M = mats[0] if len(mats) == 1 else sp.vstack(mats, format="csr")
        assert M.shape[1] == len(self.vec.vocabulary_), f"{tag}: transform produced {M.shape[1]} columns, vocab has {len(self.vec.vocabulary_)}"
        return M

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True); self.pool = None


# ------------------------------------------------------------------ per-country blocking
def block_country(country: str, q: pl.DataFrame, p: pl.DataFrame, pairs: pl.DataFrame | None, cfg: BlockingConfig,
                  alias: dict[str, str], out_dir, gpu: bool) -> dict:
    t0 = time.time()
    q = finalize_addresses(q, alias, f"{country} S1", cfg)
    p = finalize_addresses(p, alias, f"{country} S2/S3", cfg)
    q_ids = np.asarray(q["entity_id"].to_list(), dtype=object); p_ids = np.asarray(p["entity_id"].to_list(), dtype=object)
    q_name = np.asarray(q["name_core"].to_list(), dtype=object); p_name = np.asarray(p["name_core"].to_list(), dtype=object)
    q_addr = np.asarray(q["addr_norm"].to_list(), dtype=object); p_addr = np.asarray(p["addr_norm"].to_list(), dtype=object)
    q_state = q["state_norm"].to_list(); p_state = p["state_norm"].to_list()
    q_st = np.asarray([s or "" for s in q_state]); p_st = np.asarray([s or "" for s in p_state])
    final_q = q.select("entity_id", "addr_norm", "state_norm"); final_p = p.select("entity_id", "addr_norm", "state_norm")
    del q, p
    q_na = np.asarray([f"{a} {b}".strip() for a, b in zip(q_name, q_addr)], dtype=object)
    p_na = np.asarray([f"{a} {b}".strip() for a, b in zip(p_name, p_addr)], dtype=object)
    shard = bool(cfg.shard_by_state and alias)
    groups = plan_groups(q_state, p_state, shard)
    log(f"[{country}] {len(q_ids):,} S1, {len(p_ids):,} pool rows, {len(groups)} shard(s)"
        + (f" (pool-unknown {int((p_st == '').sum()):,} rows in every shard, query-unknown {int((q_st == '').sum()):,})" if shard else ""))
    if pairs is not None:
        pairs = pairs.with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))

    # fit vectorizers once per country
    transformers: dict[str, Transformer] = {}
    if cfg.word.enabled:
        transformers["word"] = Transformer(fit_vectorizer("word", p_na.tolist(), cfg.word_max_df, cfg.char_fit_sample, cfg.seed, f"{country}/word"), cfg.n_workers(), cfg.transform_chunk)
    if cfg.char_name.enabled:
        transformers["char_name"] = Transformer(fit_vectorizer("char", p_name.tolist(), 1.0, cfg.char_fit_sample, cfg.seed, f"{country}/char_name"), cfg.n_workers(), cfg.transform_chunk)
    if cfg.char_addr.enabled:
        transformers["char_addr"] = Transformer(fit_vectorizer("char", p_addr.tolist(), 1.0, cfg.char_fit_sample, cfg.seed, f"{country}/char_addr"), cfg.n_workers(), cfg.transform_chunk)
    texts = {"word": (q_na, p_na), "char_name": (q_name, p_name), "char_addr": (q_addr, p_addr)}
    bcfg = {"word": cfg.word, "char_name": cfg.char_name, "char_addr": cfg.char_addr}

    stats = {b: 0 for b in BLOCKER_NAMES}; n_pairs = 0; n_pos = 0; parts_dir = candidate_parts_dir(out_dir)
    hits_dir = out_dir / "hits" / safe_name(country)
    if cfg.keep_hits:
        hits_dir.mkdir(parents=True, exist_ok=True)
    # B0 is a dict lookup, so it runs country-wide (not sharded): exact core-name matches across state shards are kept.
    exact_all = exact_blocker(q_name.tolist(), p_name.tolist(), cfg.exact_block_cap) if cfg.exact_enabled else None
    bar = pbar(groups, desc=f"shards[{country}]", unit="shard")
    for g, q_sel, p_sel in bar:
        bar.set_postfix_str(f"{g}: {len(q_sel):,} q x {len(p_sel):,} pool")
        hits: dict[str, pl.DataFrame] = {}
        if exact_all is not None:
            in_group = np.zeros(len(q_ids), dtype=bool); in_group[q_sel] = True
            hits["exact"] = exact_all.filter(pl.Series(in_group[exact_all["q_idx"].to_numpy()])) if exact_all.height else exact_all
        for name, tr in transformers.items():
            qt, pt = texts[name]
            A = tr(qt[q_sel], f"{country}/{name}/{g} q"); B = tr(pt[p_sel], f"{country}/{name}/{g} pool")
            bc = bcfg[name]; tag = f"{country}/{name}/{g}"
            if name != "word" and gpu:
                from .gpu_topk import gpu_topk
                qi, pi, sc = gpu_topk(A, B, bc.k, bc.min_cos, cfg.gpu_pool_block_rows, cfg.gpu_out_budget_bytes, cfg.gpu_query_chunk_max, tag)
            else:
                qi, pi, sc = cpu_topk(A, B, bc.k, bc.min_cos, cfg.chunk_size, cfg.n_threads(), tag)
            hits[name] = hits_frame(qi, pi, sc, bc.k, q_sel, p_sel)
            del A, B
        if cfg.dense.enabled:
            from .dense import dense_blocker
            h = dense_blocker(q_name[q_sel].tolist(), p_name[p_sel].tolist(), cfg.dense, f"{country}/{g}", None)
            if h.height:
                h = h.with_columns(pl.Series("q_idx", q_sel[h["q_idx"].to_numpy()]), pl.Series("p_idx", p_sel[h["p_idx"].to_numpy()]))
            hits["dense"] = h
        for b, h in hits.items():
            stats[b] += int(h.height)
            if cfg.keep_hits:
                h.write_parquet(hits_dir / f"{b}__{g}.parquet")
        # ---- union + provenance + labels -> one parquet part per shard
        u = rrf_union(hits, cfg.rrf_weights, cfg.rrf_k, cfg.cap)
        del hits
        if u.height == 0:
            continue
        qi = u["q_idx"].to_numpy(); pi = u["p_idx"].to_numpy()
        sm = np.where((q_st[qi] == "") | (p_st[pi] == ""), "unknown", np.where(q_st[qi] == p_st[pi], "same", "diff"))
        cand = p_ids[pi]
        u = u.with_columns(
            pl.Series("s1_id", q_ids[qi].tolist()), pl.Series("cand_id", cand.tolist()),
            pl.Series("src", [x[:2] for x in cand]), pl.lit(country).alias("country"), pl.Series("state_match", sm.tolist()),
        ).drop(["q_idx", "p_idx"]).select(CAND_COLS)
        if pairs is not None:
            u = u.join(pairs, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
            n_pos += int(u["label"].sum())
        u.write_parquet(parts_dir / f"{safe_name(country)}__{g}.parquet")
        n_pairs += u.height
    for tr in transformers.values():
        tr.close()
    summary = {"country": country, "n_s1": int(len(q_ids)), "n_pool": int(len(p_ids)), "n_shards": len(groups), "sharded": shard,
               "gpu": gpu, "hits": stats, "n_pairs": n_pairs, "cands_per_s1_mean": n_pairs / max(len(q_ids), 1),
               "elapsed_s": round(time.time() - t0, 1)}
    if pairs is not None:
        summary["positives_retrieved"] = n_pos; summary["gt_pairs"] = int(pairs.height); summary["pair_recall"] = n_pos / max(pairs.height, 1)
    log(f"[{country}] done: {summary}")
    return {"summary": summary, "vectorizers": {k: v.vec for k, v in transformers.items()}, "final_q": final_q, "final_p": final_p}


# ------------------------------------------------------------------ stage
def stage_block(cfg: BlockingConfig, split: str) -> None:
    out = cfg.split_dir(split)
    parts_dir = candidate_parts_dir(out)
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    for old in out.glob("candidates_*.pkl"):
        old.unlink()
    if (out / "hits").exists():
        shutil.rmtree(out / "hits")
    alias_path = cfg.split_dir("train") / "aliases.pkl"
    tables = load_pkl(alias_path) if alias_path.exists() else {}
    if not tables:
        log("WARNING: no aliases.pkl found (run the train `aliases` stage first); proceeding without state aliases / sharding")
    countries = list_countries(out)
    log(f"countries in {split}: {countries}")
    pairs_all = gt_to_pairs(read_ground_truth(cfg.data_dir / "train" / "train_ground_truth.tsv")) if split == "train" else None
    gpu = use_gpu(cfg)
    sampling = cfg.sample_s1 is not None or cfg.pool_fraction < 1.0
    sample_ids = {"s1_ids": [], "pool_ids": []}
    summaries, vecs, final_q, final_p = [], {}, [], []
    for country in countries:
        with stage(f"block [{split}/{country}]"):
            q, p = load_country(out, country)
            pairs_c = pairs_all.join(q.select(pl.col("entity_id").alias("s1_id")), on="s1_id") if pairs_all is not None else None
            if cfg.sample_s1 is not None:
                q, p, pairs_c = subsample_country(q, p, pairs_c, cfg, len(countries))
            elif cfg.pool_fraction < 1.0:
                p = p.sample(fraction=cfg.pool_fraction, seed=cfg.seed)
            if sampling:
                sample_ids["s1_ids"].extend(q["entity_id"].to_list()); sample_ids["pool_ids"].extend(p["entity_id"].to_list())
            if q.height == 0 or p.height == 0:
                log(f"[{country}] skipped: {q.height} S1 rows, {p.height} pool rows"); continue
            alias = (tables.get(country) or {}).get("alias", {})
            info = block_country(country, q, p, pairs_c, cfg, alias, out, gpu)
            summaries.append(info["summary"]); vecs[country] = info["vectorizers"]
            final_q.append(info["final_q"]); final_p.append(info["final_p"])
    if not candidate_parts(out):
        raise RuntimeError("no candidates produced")
    with stage(f"assemble [{split}]"):
        index = write_pkl_parts(out, cfg)
        save_pkl(pl.concat(final_q), out / "final_addr_source1.pkl")
        save_pkl(pl.concat(final_p), out / "final_addr_source23.pkl")
        save_pkl({"vectorizers": vecs, "config": cfg.to_dict()}, out / "vectorizers.pkl")
        n_pairs = sum(s["n_pairs"] for s in summaries); n_s1 = sum(s["n_s1"] for s in summaries); n_pool = sum(s["n_pool"] for s in summaries)
        save_json({"split": split, "config": cfg.to_dict(), "gpu": gpu, "countries": summaries, "n_pairs": n_pairs, "n_s1": n_s1, "n_pool": n_pool,
                   "cands_per_s1_mean": n_pairs / max(n_s1, 1), "pkl_parts": index}, out / "block_summary.json")
        sample_path = out / "sample_ids.pkl"
        if sampling:
            save_pkl(sample_ids, sample_path)
        elif sample_path.exists():
            sample_path.unlink(); log("removed stale sample_ids.pkl from an earlier --sample run")
        if gpu:
            from .gpu_topk import gpu_memory_summary
            log(f"GPU memory: {gpu_memory_summary()}")


def write_pkl_parts(split_dir, cfg: BlockingConfig) -> dict:
    """Group the per-shard parquet parts into pkl parts of <= pkl_part_rows rows (pandas, pyarrow-backed columns)."""
    files = candidate_parts(split_dir)
    parts, buf, n_buf, k = [], [], 0, 0

    def flush():
        nonlocal buf, n_buf, k
        if not buf:
            return
        df = pl.concat(buf)
        name = f"candidates_{k:02d}.pkl"
        save_pkl(df, split_dir / name)
        parts.append({"file": name, "rows": int(df.height)})
        buf, n_buf, k = [], 0, k + 1

    for f in pbar(files, desc="pkl parts", unit="part"):
        df = pl.read_parquet(f)
        buf.append(df); n_buf += df.height
        if n_buf >= cfg.pkl_part_rows:
            flush()
    flush()
    index = {"parts": parts, "rows": sum(p["rows"] for p in parts), "columns": CAND_COLS + (["label"] if files and "label" in pl.read_parquet_schema(files[0]) else [])}
    save_json(index, split_dir / "candidates_index.json")
    if not cfg.export_parquet:
        shutil.rmtree(candidate_parts_dir(split_dir))
    return index
