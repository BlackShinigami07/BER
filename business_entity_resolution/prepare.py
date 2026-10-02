"""Stage `prepare`: read the three source TSVs of a split, normalise names/addresses (process pool), write
normalized_*.pkl (ML-facing), per-country parquet parts (pipeline-internal), suffix_vocab.pkl (unsupervised) and,
for train, splits.pkl (ml_train / ml_valid by S1 entity)."""
from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from .aliases import learn_suffix_vocab
from .config import BlockingConfig
from .io import SOURCES, normalized_part, read_source, save_countries, save_pkl, source_path
from .normalize import NameNormalizer, basic_clean, components_to_string, script_class, split_components
from .progress import log, pbar

_NAME_NORM: NameNormalizer | None = None
PART_COLS = ["entity_id", "country", "name_core", "name_full", "addr_comps", "name_script"]


def _init_worker(suffixes: list[str]):
    global _NAME_NORM
    _NAME_NORM = NameNormalizer(suffixes=set(suffixes))


def _norm_chunk(args):
    names, addrs = args
    nn = _NAME_NORM or NameNormalizer()
    full, core, comps, scripts = [], [], [], []
    for n in names:
        f, c = nn(n)
        full.append(f); core.append(c); scripts.append(script_class(n))
    for a in addrs:
        comps.append(components_to_string(split_components(a)))
    return full, core, comps, scripts


def _count_tokens_chunk(names):
    """(all-token counts, last-token counts) for a chunk of raw names."""
    c: collections.Counter = collections.Counter()
    last: collections.Counter = collections.Counter()
    for n in names:
        toks = basic_clean(n).split()
        c.update(toks)
        if toks:
            last[toks[-1]] += 1
    return c, last


def _run_chunks(fn, payloads, workers: int, desc: str, init=None, initargs=()):
    if workers <= 1 or len(payloads) <= 1:
        if init:
            init(*initargs)
        return [fn(p) for p in pbar(payloads, desc=desc, unit="chunk", leave=False)]
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads)), initializer=init, initargs=initargs) as ex:
        return list(pbar(ex.map(fn, payloads), total=len(payloads), desc=desc, unit="chunk", leave=False))


def learn_split_suffixes(frames: dict[str, pl.DataFrame], cfg: BlockingConfig) -> set[str]:
    def counts(df):
        chunks = [df["business_name"].slice(i, cfg.norm_chunk_size).to_list() for i in range(0, df.height, cfg.norm_chunk_size)]
        tot: collections.Counter = collections.Counter()
        last: collections.Counter = collections.Counter()
        for c, l in _run_chunks(_count_tokens_chunk, chunks, cfg.n_workers(), "suffix vocab: token counts"):
            tot.update(c); last.update(l)
        return tot, last
    pool, pool_last = counts(pl.concat([frames["source2"].select("business_name"), frames["source3"].select("business_name")]))
    s1, _ = counts(frames["source1"].select("business_name"))
    vocab = learn_suffix_vocab(pool, pool_last, s1)
    log(f"learned suffix vocabulary: {len(vocab)} tokens, e.g. {sorted(vocab)[:25]}")
    return vocab


def normalize_frame(df: pl.DataFrame, suffixes: set[str], cfg: BlockingConfig, tag: str) -> pl.DataFrame:
    payloads = []
    for i in range(0, df.height, cfg.norm_chunk_size):
        sl = df.slice(i, cfg.norm_chunk_size)
        payloads.append((sl["business_name"].to_list(), sl["business_address"].to_list()))
    parts = _run_chunks(_norm_chunk, payloads, cfg.n_workers(), f"normalize {tag}", _init_worker, (sorted(suffixes),))
    cols = {"name_full": [], "name_core": [], "addr_comps": [], "name_script": []}
    for p in parts:
        cols["name_full"].extend(p[0]); cols["name_core"].extend(p[1]); cols["addr_comps"].extend(p[2]); cols["name_script"].extend(p[3])
    del parts
    return df.with_columns([pl.Series(k, v) for k, v in cols.items()])


def make_splits(s1: pl.DataFrame, cfg: BlockingConfig) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    parts = []
    for country in sorted(s1["country"].unique().to_list()):
        ids = s1.filter(pl.col("country") == country)["entity_id"].to_list()
        r = rng.random(len(ids))
        parts.append(pl.DataFrame({"entity_id": ids, "fold": np.where(r < cfg.valid_fraction, "ml_valid", "ml_train")}))
    sp = pl.concat(parts)
    log("splits: " + str(sp["fold"].value_counts().sort("fold").to_dict(as_series=False)))
    return sp


def stage_prepare(cfg: BlockingConfig, split: str) -> None:
    out = cfg.split_dir(split)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    frames = {s: read_source(source_path(cfg.data_dir, split, s)) for s in SOURCES}
    for s, df in frames.items():
        if df["entity_id"].n_unique() != df.height:
            raise ValueError(f"{split}/{s}: duplicate entity_id values")
    suffixes = learn_split_suffixes(frames, cfg)
    save_pkl({"suffixes": sorted(suffixes)}, out / "suffix_vocab.pkl")
    countries: set[str] = set()
    for s in SOURCES:
        df = frames.pop(s)
        nd = normalize_frame(df, suffixes, cfg, f"{split}/{s}")
        del df
        cs = sorted(nd["country"].unique().to_list()); countries.update(cs)
        for c in cs:
            nd.filter(pl.col("country") == c).select(PART_COLS).write_parquet(normalized_part(out, s, c))
        log(f"{split}/{s}: {nd.height:,} rows; countries {cs}; name_script dist {nd['name_script'].value_counts().sort('name_script').to_dict(as_series=False)}")
        if split == "train" and s == "source1":
            save_pkl(make_splits(nd, cfg), out / "splits.pkl")
        save_pkl(nd, out / f"normalized_{s}.pkl")
        del nd
    save_countries(out, sorted(countries))
