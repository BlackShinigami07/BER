"""WP-2: streaming feature builder for the CatBoost stage-A classifier (CATBOOST_PLAN.md sections 1, 2, 6).

    python -m business_entity_resolution.ml gbm-features build --artifacts-dir artifacts        # train/ml/gbm_fit.parquet
    python -m business_entity_resolution.ml gbm-features build --artifacts-dir dev/mini_artifacts

How every feature is computed identically on train, valid and test (the train/test-shift rule of section 7):
1. read ALL rows of one candidate part (a state shard) and add the part-level families (candidate graph) on it:
   every S1 entity of the part, every fold - exactly what the test split sees;
2. only then restrict to the entities needed (the fit subsample, or ml_valid) and cut them into entity-complete
   chunks of ~`chunk_rows` rows, each streamed from the part file (the largest part has 15.6M rows);
3. per chunk: normalised text, the 47 base features, the entity-level families (all candidates of the entity present);
4. for the fit matrix only, the row filter of section 4 runs AFTER the features, so context features still see
   every candidate of the entity.

Fit set: 50% of the ml_train entities (country-stratified, ordered by a stable hash of the id, seed 42), 5% of them held
out for early stopping (`fold` = early_stop); rows `rrf_rank <= 20 or n_blockers >= 2` plus every positive plus a
5% hashed sample of the remaining rows with weight 20 (so the probabilities stay calibrated on the full distribution).
Text lookups are streamed from a parquet cache (ml/gbm_text_{q,c}.parquet, built once, one pickle in RAM at a time),
so no step holds the ~2.5 GB candidate-pool text in memory. Output: train/ml/gbm_fit.parquet (float32 features +
s1_id, cand_id, country, label, weight, fold), per-part resumable files in train/ml/gbm_fit_parts/, gbm_fit_config.json.
Valid / test are never stored: `iter_features` yields one entity-complete chunk at a time for ml/gbm.py predict.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from ..io import load_frame, load_json, save_json
from ..progress import available_ram_gb, fmt_secs, log, pbar, rss_gb, stage
from .common import candidate_part_files, load_folds, ml_dir
from .features import CAND_COLS, TEXT_FIELDS, FrameText, features_for, pair_features
from .features_extra import PRODUCTION_FAMILIES, ExtraLookups, add_part_families, load_extra_lookups, part_columns

PART_INPUT_COLS = ["s1_id", "cand_id", "rrf_score"]   # all a part-level family reads from the candidate table
FETCH_THREADS = 8


# ---------------------------------------------------------------- stable hashing (version-independent)
def id_number(ids: pl.Series) -> np.ndarray:
    """All digits of an id as uint64 ("S2-172049742" -> 2172049742): the source digit keeps S2/S3 ids apart."""
    return ids.str.replace_all(r"\D", "").cast(pl.UInt64, strict=False).fill_null(0).to_numpy().astype(np.uint64)


def mix64(x: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 finaliser: a deterministic, platform-independent 64-bit hash of uint64 values."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def pair_hash(s1: pl.Series, cand: pl.Series, seed: int) -> np.ndarray:
    return mix64(mix64(id_number(s1), seed) ^ id_number(cand), seed)


# ---------------------------------------------------------------- text lookups streamed from parquet
class TextStore:
    """Normalised text of one split, fetched by id from a parquet cache (built once from the pkl artifacts)."""

    SIDES = {"q": (["normalized_source1.pkl"], "final_addr_source1.pkl", "s1_id"),
             "c": (["normalized_source2.pkl", "normalized_source3.pkl"], "final_addr_source23.pkl", "cand_id")}

    def __init__(self, split_dir, cache_dir=None):
        self.split_dir = Path(split_dir)
        cache_dir = Path(cache_dir) if cache_dir else self.split_dir / "ml"
        self.files = {s: cache_dir / f"gbm_text_{s}.parquet" for s in self.SIDES}
        for side in self.SIDES:
            if not self._fresh(side):
                self._build(side)

    def _fresh(self, side: str) -> bool:
        f = self.files[side]
        norm, addr, _ = self.SIDES[side]
        src = [self.split_dir / x for x in norm + [addr]]
        return f.exists() and all(f.stat().st_mtime >= s.stat().st_mtime for s in src if s.exists())

    def _build(self, side: str) -> None:
        norm_files, addr_file, key = self.SIDES[side]
        out = self.files[side]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = [out.with_name(f"{out.stem}.tmp{i}.parquet") for i in range(len(norm_files) + 1)]
        t0 = time.time()
        for f, dst in zip(norm_files, tmp):   # one pickle in RAM at a time
            load_frame(self.split_dir / f).select(
                pl.col("entity_id").alias(key), pl.col("country").fill_null("").alias("country"),
                pl.col("name_full").fill_null("").alias(f"{side}_full"), pl.col("name_core").fill_null("").alias(f"{side}_core"),
                pl.col("addr_comps").fill_null("").alias(f"{side}_comps"), pl.col("business_address").fill_null("").alias(f"{side}_raw"),
                pl.col("name_script").fill_null("latin").alias(f"{side}_script")).write_parquet(dst)
        load_frame(self.split_dir / addr_file).select(pl.col("entity_id").alias(key), pl.col("addr_norm").fill_null("").alias(f"{side}_norm")
                                                      ).write_parquet(tmp[-1])
        cols = [key, "country"] + [f"{side}_{f}" for f in TEXT_FIELDS]
        (pl.concat([pl.scan_parquet(t) for t in tmp[:-1]])
           .join(pl.scan_parquet(tmp[-1]), on=key, how="left")
           .with_columns(pl.col(f"{side}_norm").fill_null("")).select(cols)
           .sink_parquet(out.with_suffix(".tmp"), compression="zstd"))
        os.replace(out.with_suffix(".tmp"), out)
        for t in tmp:
            t.unlink(missing_ok=True)
        log(f"text cache {out} built in {fmt_secs(time.time() - t0)}")

    def _fetch(self, side: str, ids: pl.Series, cols) -> pl.DataFrame:
        """Rows whose id is in `ids`, read row group by row group (8 threads) and filtered before concatenation:
        ~2.5 s and < 1 GB for a 1M-row chunk, where a polars semi-join / is_in scan of the 10M-row pool text peaked
        at 3-5 GB."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        key = self.SIDES[side][2]
        cols = [key] + ([f"{side}_{f}" for f in TEXT_FIELDS] if cols is None else list(cols))
        want = ids.unique().drop_nulls().to_arrow()
        path, local = str(self.files[side]), threading.local()

        def one(rg: int):
            pf = getattr(local, "pf", None)
            if pf is None:
                pf = local.pf = pq.ParquetFile(path)
            m = pc.is_in(pf.read_row_group(rg, columns=[key], use_threads=False)[key], value_set=want)
            return pf.read_row_group(rg, columns=cols, use_threads=False).filter(m) if pc.any(m).as_py() else None

        n_rg = pq.ParquetFile(path).metadata.num_row_groups
        with ThreadPoolExecutor(FETCH_THREADS) as ex:
            parts = [t for t in ex.map(one, range(n_rg)) if t is not None]
        if not parts:
            return pl.DataFrame(schema={c: pl.String for c in cols})
        return pl.from_arrow(pa.concat_tables(parts))

    def s1(self, ids: pl.Series, cols=None) -> pl.DataFrame:
        return self._fetch("q", ids, cols)

    def pool(self, ids: pl.Series, cols=None) -> pl.DataFrame:
        return self._fetch("c", ids, cols)

    pool_text = pool

    def countries(self) -> pl.DataFrame:
        return pl.read_parquet(self.files["q"], columns=["s1_id", "country"])


# ---------------------------------------------------------------- one part -> entity-complete feature chunks
def entity_slices(s1_sorted: pl.Series, chunk_rows: int) -> list[tuple[int, int]]:
    """Row ranges of ~chunk_rows that never split an entity (the rows of an entity must be contiguous)."""
    n = len(s1_sorted)
    if n == 0:
        return []
    starts = (s1_sorted != s1_sorted.shift(1)).fill_null(True).arg_true().to_numpy().astype(np.int64)
    bounds = starts[np.r_[True, np.diff(starts // max(1, chunk_rows)) > 0]]
    ends = np.r_[bounds[1:], n]
    return list(zip(bounds.tolist(), ends.tolist()))


def part_frames(part_file, text, lookups: ExtraLookups | None, *, families=PRODUCTION_FAMILIES, entities: pl.DataFrame | None = None,
                label: bool = False, chunk_rows: int = 1_000_000, timings: dict | None = None,
                per_chunk_text: int = 4_000_000) -> Iterator[pl.DataFrame]:
    """Feature frames (KEY_COLS present + BASE_FEATURES + the families' columns) for one candidate part.

    The part-level families (candidate graph) run on a slim read of EVERY row of the part (s1_id, cand_id, rrf_score).
    `entities` (s1_id [+ extra key columns such as fold]) then restricts the output. The kept entities are cut into
    entity-complete chunks of ~chunk_rows rows; each chunk is streamed from the parquet file by an s1_id filter and
    joined to its graph columns by key, so no step holds a whole part (the France shard has 15.6M rows). Text is
    fetched once per part, or per chunk when the kept rows exceed `per_chunk_text`. Row order inside a chunk is not
    meaningful (every feature is keyed by s1_id / cand_id)."""
    pcols = part_columns(families)
    graph = add_part_families(pl.read_parquet(part_file, columns=PART_INPUT_COLS), text, families, timings)
    graph = graph.select("s1_id", "cand_id", *pcols)
    if entities is not None:
        graph = graph.join(entities.select("s1_id"), on="s1_id", how="semi")
    if graph.height == 0:
        return
    sizes = graph.group_by("s1_id").len().sort("s1_id")
    sizes = sizes.with_columns((pl.col("len").cum_sum() - pl.col("len")).floordiv(max(1, chunk_rows)).alias("_chunk"))
    per_chunk = graph.height > per_chunk_text
    if not per_chunk:
        s1_text, pool_text = text.s1(graph["s1_id"]), text.pool(graph["cand_id"])
    cols = features_for(families)
    read_cols = CAND_COLS + (["label"] if label else [])
    for _, ids in sizes.select("_chunk", "s1_id").group_by("_chunk", maintain_order=True):
        want = ids.select("s1_id")
        c = (pl.scan_parquet(part_file).select(read_cols).join(want.lazy(), on="s1_id", how="semi").collect(engine="streaming")
               .join(graph.join(want, on="s1_id", how="semi"), on=["s1_id", "cand_id"], how="left"))
        if entities is not None and entities.width > 1:
            c = c.join(entities, on="s1_id", how="left")
        if per_chunk:
            s1_text, pool_text = text.s1(c["s1_id"]), text.pool(c["cand_id"])
        yield pair_features(c, s1_text, pool_text, extra_lookups=lookups, families=families, columns=cols, timings=timings)


@lru_cache(maxsize=4)
def open_split(artifacts_dir, split_name: str) -> tuple[TextStore, ExtraLookups]:
    """Text store + lookup tables of a split, loaded once per process (predict streams many parts)."""
    d = Path(artifacts_dir) / split_name
    return TextStore(d, ml_dir(artifacts_dir, split_name)), load_extra_lookups(d)


@lru_cache(maxsize=4)
def valid_entities(artifacts_dir) -> pl.DataFrame:
    return load_folds(artifacts_dir).filter(pl.col("fold") == "ml_valid").select("s1_id")


# ---------------------------------------------------------------- the fit matrix
@dataclass(frozen=True)
class BuildConfig:
    artifacts_dir: Path
    fit_share: float = 0.50             # share of ml_train entities in the fit set (section 4)
    seed: int = 42
    early_stop_share: float = 0.05      # of the fit entities, by entity
    rrf_rank_keep: int = 20             # row filter: rrf_rank <= 20 or n_blockers >= 2 (+ every positive)
    min_blockers_keep: int = 2
    tail_keep: float = 0.05             # hashed sample of the filtered-out rows ...
    tail_weight: float = 20.0           # ... with this weight
    chunk_rows: int = 1_000_000
    families: tuple[str, ...] = field(default=PRODUCTION_FAMILIES)

    def fingerprint(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("artifacts_dir", "chunk_rows")}
        d["families"] = list(self.families)
        d["features"] = features_for(self.families)
        return d


def selection(cfg: BuildConfig, text: TextStore | None = None) -> pl.DataFrame:
    """Country-stratified fit subsample of ml_train (the first fit_share of each country by a stable hash of the id)
    and its early-stopping holdout (by a second hash). Stored in train/ml/gbm_fit_entities.parquet."""
    out = ml_dir(cfg.artifacts_dir, "train") / "gbm_fit_entities.parquet"
    ent = load_folds(cfg.artifacts_dir).filter(pl.col("fold") == "ml_train").select("s1_id")
    text = text or TextStore(cfg.artifacts_dir / "train", ml_dir(cfg.artifacts_dir, "train"))
    ent = ent.join(text.countries(), on="s1_id", how="left").with_columns(pl.col("country").fill_null(""))
    ent = ent.with_columns(pl.Series("_h", mix64(id_number(ent["s1_id"]), cfg.seed)),
                           pl.Series("_h2", mix64(id_number(ent["s1_id"]), cfg.seed + 1)))
    ent = ent.sort(["country", "_h", "s1_id"]).with_columns(pl.int_range(pl.len()).over("country").alias("_i"),
                                                              pl.len().over("country").alias("_n"))
    es_mod = max(1, int(round(1 / cfg.early_stop_share)))
    sel = (ent.filter(pl.col("_i") < (pl.col("_n") * cfg.fit_share).round())
              .with_columns(pl.when(pl.col("_h2") % es_mod == 0).then(pl.lit("early_stop")).otherwise(pl.lit("fit")).alias("fold"))
              .select("s1_id", "country", "fold").sort("s1_id"))
    out.parent.mkdir(parents=True, exist_ok=True)
    sel.write_parquet(out)
    by = {f"{c}/{f}": int(n) for c, f, n in sel.group_by("country", "fold").len().sort("country", "fold").iter_rows()}
    log(f"fit entities: {sel.height:,} of {ent.height:,} ml_train ({sel.height / max(1, ent.height):.1%}, country-stratified) {by} -> {out}")
    return sel


def row_filter(cfg: BuildConfig) -> tuple[pl.Expr, pl.Expr]:
    """(main rule of section 0, main rule + every positive)."""
    main = (pl.col("rrf_rank") <= cfg.rrf_rank_keep) | (pl.col("n_blockers") >= cfg.min_blockers_keep)
    return main, main | (pl.col("label") == 1)


def _write_frames(frames: Iterator[pl.DataFrame], path: Path) -> int:
    import pyarrow.parquet as pq
    tmp, writer, n = path.with_suffix(".tmp"), None, 0
    try:
        for f in frames:
            if f.height == 0:
                continue
            t = f.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            writer.write_table(t)
            n += f.height
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        return 0
    os.replace(tmp, path)
    return n


def build_fit(cfg: BuildConfig, *, resume: bool = True) -> Path:
    """Write train/ml/gbm_fit.parquet; per-part files make the build resumable (a changed config rebuilds)."""
    t0 = time.time()
    out_dir = ml_dir(cfg.artifacts_dir, "train")
    fit_path, meta_path, cache = out_dir / "gbm_fit.parquet", out_dir / "gbm_fit_config.json", out_dir / "gbm_fit_parts"
    fp = cfg.fingerprint()
    old = load_json(meta_path) if meta_path.exists() else {}
    if old.get("fingerprint") != fp or not resume:
        for f in cache.glob("*.parquet"):
            f.unlink()
        fit_path.unlink(missing_ok=True)
    elif fit_path.exists() and old.get("done"):
        log(f"{fit_path} is up to date (same config); use --no-resume to rebuild")
        return fit_path
    cache.mkdir(parents=True, exist_ok=True)
    save_json({"fingerprint": fp, "done": False}, meta_path)
    split = cfg.artifacts_dir / "train"
    text, lookups = open_split(cfg.artifacts_dir, "train")
    sel = selection(cfg, text)
    main, keep_rule = row_filter(cfg)
    tail_mod = max(1, int(round(1 / cfg.tail_keep)))
    timings: dict[str, float] = {}
    stats: dict[str, dict] = {}
    for part in pbar(candidate_part_files(split), desc="gbm fit features", unit="part"):
        out_part = cache / part.name
        st_file = out_part.with_suffix(".json")
        if out_part.exists() and st_file.exists():
            stats[part.name] = load_json(st_file)
            continue
        t_part, st = time.time(), {"all_rows": 0, "all_pos": 0, "main_rows": 0, "main_pos": 0, "rows": 0, "pos": 0, "tail_rows": 0}

        def kept(frames):
            for f in frames:
                tail_hit = pl.Series(pair_hash(f["s1_id"], f["cand_id"], cfg.seed) % np.uint64(tail_mod) == 0)
                f = f.with_columns(tail_hit.alias("_tail"))
                m = f.select(main.alias("m"), keep_rule.alias("k"), pl.col("label"))
                st["all_rows"] += f.height
                st["all_pos"] += int(m["label"].sum())
                st["main_rows"] += int(m["m"].sum())
                st["main_pos"] += int((m["m"] & (m["label"] == 1)).sum())
                f = (f.filter(keep_rule | pl.col("_tail"))
                      .with_columns(pl.when(keep_rule).then(1.0).otherwise(cfg.tail_weight).cast(pl.Float32).alias("weight"),
                                    pl.col("label").cast(pl.Int8)))
                st["rows"] += f.height
                st["pos"] += int(f["label"].sum())
                st["tail_rows"] += int((f["weight"] > 1).sum())
                yield f.select("s1_id", "cand_id", "country", "label", "weight", "fold", *features_for(cfg.families))

        _write_frames(kept(part_frames(part, text, lookups, families=cfg.families, entities=sel.select("s1_id", "fold"),
                                       label=True, chunk_rows=cfg.chunk_rows, timings=timings)), out_part)
        st.update(seconds=round(time.time() - t_part, 1), peak_rss_gb=rss_gb())
        save_json(st, st_file)
        stats[part.name] = st
        log(f"[{part.name}] kept {st['rows']:,}/{st['all_rows']:,} rows, {st['pos']:,}/{st['all_pos']:,} positives in "
            f"{st['seconds']:.0f}s; peak RSS {st['peak_rss_gb'] or float('nan'):.2f} GB, available {available_ram_gb() or float('nan'):.1f} GB")

    parts = sorted(cache.glob("*.parquet"))
    if not parts:
        raise RuntimeError("no fit rows were written (empty selection?)")
    with stage(f"merge {len(parts)} parts -> {fit_path.name}"):
        pl.scan_parquet(parts).sink_parquet(fit_path.with_suffix(".tmp"), compression="zstd", row_group_size=500_000)
        os.replace(fit_path.with_suffix(".tmp"), fit_path)
    tot = {k: sum(s.get(k, 0) for s in stats.values()) for k in ("all_rows", "all_pos", "main_rows", "main_pos", "rows", "pos", "tail_rows")}
    n_feat = len(features_for(cfg.families))
    summary = {
        "fit_entities": sel.height, "early_stop_entities": int((sel["fold"] == "early_stop").sum()),
        "rows_of_fit_entities": tot["all_rows"], "positives_of_fit_entities": tot["all_pos"],
        "rule_row_share": tot["main_rows"] / max(1, tot["all_rows"]), "rule_positive_share": tot["main_pos"] / max(1, tot["all_pos"]),
        "fit_rows": tot["rows"], "fit_positives": tot["pos"], "tail_rows": tot["tail_rows"],
        "fit_row_share": tot["rows"] / max(1, tot["all_rows"]), "fit_positive_share": tot["pos"] / max(1, tot["all_pos"]),
        "n_features": n_feat, "matrix_gb_float32": tot["rows"] * n_feat * 4 / 2**30,
        "file_gb": fit_path.stat().st_size / 2**30, "seconds": round(time.time() - t0, 1), "peak_rss_gb": rss_gb(),
        "feature_seconds": {k: round(v, 1) for k, v in sorted(timings.items(), key=lambda kv: -kv[1])},
    }
    save_json({"fingerprint": fp, "done": True, "config": {**asdict(cfg), "artifacts_dir": str(cfg.artifacts_dir)},
               "summary": summary, "parts": stats}, meta_path)
    log(f"fit matrix: {tot['rows']:,} rows x {n_feat} features = {summary['matrix_gb_float32']:.2f} GB float32 "
        f"({summary['file_gb']:.2f} GB on disk) from {sel.height:,} entities; rule keeps {summary['rule_row_share']:.1%} of rows / "
        f"{summary['rule_positive_share']:.2%} of positives; with every positive + the {cfg.tail_keep:.0%} weighted tail: "
        f"{summary['fit_row_share']:.1%} / {summary['fit_positive_share']:.2%}; peak RSS {summary['peak_rss_gb'] or float('nan'):.2f} GB; "
        f"{fmt_secs(summary['seconds'])}")
    log("feature seconds by family: " + json.dumps(summary["feature_seconds"]))
    return fit_path


# ---------------------------------------------------------------- valid / test streaming (never stored)
def iter_feature_frames(artifacts_dir, split: str, *, families=PRODUCTION_FAMILIES, chunk_rows: int = 1_000_000,
                        timings: dict | None = None, parts: list[Path] | None = None) -> Iterator[tuple[str, pl.DataFrame]]:
    """(part name, feature frame) per entity-complete chunk. split=valid: every ml_valid entity of the train
    candidates (with `label`); split=test: every test candidate. Never concatenates parts."""
    if split not in ("valid", "test"):
        raise ValueError("split must be 'valid' or 'test'")
    src = "train" if split == "valid" else "test"
    text, lookups = open_split(artifacts_dir, src)
    ents = valid_entities(artifacts_dir) if split == "valid" else None
    for part in parts or candidate_part_files(Path(artifacts_dir) / src):
        for f in part_frames(part, text, lookups, families=families, entities=ents, label=split == "valid",
                             chunk_rows=chunk_rows, timings=timings):
            yield part.name, f


def iter_features(artifacts_dir, split: str, *, families=PRODUCTION_FAMILIES) -> Iterator[tuple[pl.DataFrame, np.ndarray]]:
    """The WP-2 streaming contract: (keys, float32 feature matrix) per entity-complete chunk of each part."""
    cols = features_for(families)
    for _, f in iter_feature_frames(artifacts_dir, split, families=families):
        yield f.select([k for k in ("s1_id", "cand_id", "country", "label") if k in f.columns]), \
              f.select(cols).to_numpy().astype(np.float32, copy=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="train/ml/gbm_fit.parquet")
    b.add_argument("--artifacts-dir", default="artifacts")
    b.add_argument("--fit-share", type=float, default=0.50)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--tail-keep", type=float, default=0.05)
    b.add_argument("--tail-weight", type=float, default=20.0)
    b.add_argument("--chunk-rows", type=int, default=1_000_000)
    b.add_argument("--no-resume", action="store_true")
    t = sub.add_parser("text", help="only build the parquet text caches of both splits")
    t.add_argument("--artifacts-dir", default="artifacts")
    a = p.parse_args(argv)
    if a.cmd == "text":
        for s in ("train", "test"):
            if (Path(a.artifacts_dir) / s).exists():
                TextStore(Path(a.artifacts_dir) / s, ml_dir(a.artifacts_dir, s))
        return
    if not 0 < a.fit_share <= 1 or not 0 < a.tail_keep <= 1:
        raise SystemExit("--fit-share and --tail-keep must be in (0, 1]")
    with stage("gbm features: fit matrix"):
        build_fit(BuildConfig(Path(a.artifacts_dir), fit_share=a.fit_share, seed=a.seed, tail_keep=a.tail_keep,
                              tail_weight=a.tail_weight, chunk_rows=a.chunk_rows), resume=not a.no_resume)


if __name__ == "__main__":
    main()
