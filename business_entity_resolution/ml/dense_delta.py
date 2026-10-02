"""Dense-recall delta (DENSE_RECALL_PLAN.md): new India candidates from the dense job -> CatBoost -> gate -> LLM blocks.

    uv run python -m business_entity_resolution.ml dense-delta features --split train --hits llm_out/<dense job>
    uv run python -m business_entity_resolution.ml dense-delta features --split test  --hits llm_out/<dense job>
    uv run python -m business_entity_resolution.ml dense-delta gate
    uv run python -m business_entity_resolution.ml dense-delta blocks --p-min 0.05

features: pairs of dense_hits_<split>_r*.parquet that are not in the candidate parts are new. They are written as rows
with dense-only provenance (n_blockers 1, rank_dense/cos_dense, rrf_rank 60 + rank) to candidate_parts/India__zz_dense
(so export, blocks and candidate_pairs.tsv see them). CatBoost features are computed on a temporary part holding the
touched entities' complete candidate lists (old rows + new rows), and CatBoost p of the NEW rows only is written to
<split>/ml/gbm_{valid,test}_dense.parquet. Old rows keep their probabilities.
gate: ml_valid F0.5 with and without the dense rows (CatBoost rule on all ml_valid, blend rule on valid_llm).
blocks: touched test entities (a new candidate with CatBoost p >= --p-min) get a new block = their scored block members
(llm_scores_test.parquet) + the new candidates, capped at k_max by CatBoost p -> blocks/test_dense.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..io import save_json
from ..progress import log, pbar
from .common import candidate_part_files, load_folds, ml_dir

DENSE_PART = "India__zz_dense.parquet"
PART_SCHEMA = {"s1_id": pl.String, "cand_id": pl.String, "src": pl.String, "country": pl.String, "state_match": pl.String,
               "n_blockers": pl.Int8, "rrf_score": pl.Float32, "rrf_rank": pl.Int32, "rank_exact": pl.Int32, "rank_word": pl.Int32,
               "rank_char_name": pl.Int32, "rank_char_addr": pl.Int32, "rank_dense": pl.Int32, "cos_exact": pl.Float32,
               "cos_word": pl.Float32, "cos_char_name": pl.Float32, "cos_char_addr": pl.Float32, "cos_dense": pl.Float32}


def _truth(art: Path) -> pl.DataFrame:
    gt = pl.read_csv(art.parent / "dataset" / "train" / "train_ground_truth.tsv", separator="\t", quote_char=None,
                     schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("cand_id"))
              .filter(pl.col("cand_id").is_not_null() & (pl.col("cand_id") != "")))


def addr_score(pairs: pl.DataFrame, dense_in: Path, split: str) -> pl.DataFrame:
    """rapidfuzz token_set_ratio of the raw addresses (from the dense inputs) for every pair."""
    from rapidfuzz import fuzz
    tag = "train" if split == "train" else "test"
    addr = lambda f, c: pl.read_parquet(dense_in / f).select(pl.col("id").alias(c), pl.col("text").str.split(" | ").list.last().str.to_lowercase().alias(f"a_{c}"))
    j = pairs.join(addr(f"q_{tag}.parquet", "s1_id"), on="s1_id", how="left").join(addr(f"p_{tag}.parquet", "cand_id"), on="cand_id", how="left")
    a, b = j["a_s1_id"].fill_null("").to_list(), j["a_cand_id"].fill_null("").to_list()
    return j.drop("a_s1_id", "a_cand_id").with_columns(pl.Series("addr_sim", [fuzz.token_set_ratio(x, y) if y else -1.0 for x, y in zip(a, b)], pl.Float32))


def features(art: Path, split: str, hits_dir: Path, chunk_rows: int = 1_000_000, max_rank: int = 20, min_cos: float = 0.0,
             min_addr: float = -1.0, dense_in: Path = Path("dense_in"), hits_tags: tuple[str, ...] = (), part_name: str = DENSE_PART,
             out_name: str = "") -> Path:
    from .gbm import _load_models, apply_calibration, model_manifest
    from .gbm_features import open_split, part_frames
    t0 = time.time()
    tag = "train" if split == "train" else "test"
    files = sorted(f for t in (hits_tags or (tag,)) for f in Path(hits_dir).rglob(f"dense_hits_{t}_r*.parquet"))
    if not files:
        raise SystemExit(f"no dense_hits_{tag}_r*.parquet under {hits_dir}")
    hits = pl.concat([pl.read_parquet(f) for f in files]).unique(["s1_id", "cand_id"])
    parts_dir = art / split / "candidate_parts"
    india = [f for f in candidate_part_files(art / split) if "zz_dense" not in f.name]
    old_keys = pl.concat([pl.scan_parquet(f).select("s1_id", "cand_id", "country") for f in india]).join(
        hits.lazy().select("s1_id").unique(), on="s1_id", how="semi").collect(engine="streaming")
    ctry = old_keys.select("s1_id", pl.col("country").alias("_ctry")).unique("s1_id")
    new = hits.join(old_keys.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
    n_all = new.height
    new = new.filter((pl.col("rank") <= max_rank) & (pl.col("cos") >= min_cos))
    if min_addr > -1:
        new = addr_score(new, dense_in, split)
        new = new.filter((pl.col("addr_sim") >= min_addr) | (pl.col("addr_sim") < 0)).drop("addr_sim")
    log(f"[{split}] pre-filter rank <= {max_rank}, cos >= {min_cos}, addr_sim >= {min_addr} (or empty address): {n_all:,} -> {new.height:,} new pairs")
    log(f"[{split}] {hits.height:,} dense hits for {hits['s1_id'].n_unique():,} entities; {new.height:,} new pairs "
        f"for {new['s1_id'].n_unique():,} entities")
    new_rows = (new.join(ctry, on="s1_id", how="left").with_columns(
        pl.col("cand_id").str.slice(0, 2).alias("src"), pl.col("_ctry").alias("country"), pl.lit("unknown").alias("state_match"),
        pl.lit(1, pl.Int8).alias("n_blockers"), (0.5 / (60 + pl.col("rank"))).cast(pl.Float32).alias("rrf_score"),
        (60 + pl.col("rank")).cast(pl.Int32).alias("rrf_rank"), pl.col("rank").cast(pl.Int32).alias("rank_dense"),
        pl.col("cos").cast(pl.Float32).alias("cos_dense"))
        .with_columns([pl.lit(None, PART_SCHEMA[c]).alias(c) for c in ("rank_exact", "rank_word", "rank_char_name", "rank_char_addr",
                                                                      "cos_exact", "cos_word", "cos_char_name", "cos_char_addr")])
        .select([pl.col(c).cast(t) for c, t in PART_SCHEMA.items()]))
    labelled = split == "train"
    if labelled:
        tr = _truth(art).with_columns(pl.lit(1, pl.Int8).alias("label"))
        new_rows = new_rows.join(tr, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        log(f"[{split}] new pairs that are true matches: {int(new_rows['label'].sum()):,} ({new_rows['label'].mean():.2%})")
    new_rows.write_parquet(parts_dir / part_name)
    # temporary part: complete candidate lists of the touched entities (old rows + new rows) for context features
    touched = new_rows.select("s1_id").unique()
    cols = list(PART_SCHEMA) + (["label"] if labelled else [])
    old_rows = pl.concat([pl.scan_parquet(f).select(cols).join(touched.lazy(), on="s1_id", how="semi") for f in india]).collect(engine="streaming")
    tmp = ml_dir(art, split) / f"dense_tmp_{part_name}"
    pl.concat([old_rows, new_rows.select(cols)]).sort("s1_id").write_parquet(tmp)
    log(f"[{split}] feature part: {old_rows.height:,} old + {new_rows.height:,} new rows for {touched.height:,} entities")
    man = model_manifest(art)
    feats, fams = man["features"], tuple(man["fit_fingerprint"]["families"])
    models = _load_models(man, art)
    text, lookups = open_split(art, split)
    new_keys = new_rows.select("s1_id", "cand_id").with_columns(pl.lit(True).alias("_new"))
    out = []
    for fr in pbar(part_frames(tmp, text, lookups, families=fams, label=labelled, chunk_rows=chunk_rows), desc=f"dense features {split}", unit="chunk"):
        fr = fr.join(new_keys, on=["s1_id", "cand_id"], how="inner")
        if fr.height == 0:
            continue
        X = fr.select(feats).to_numpy().astype(np.float32, copy=False)
        p = apply_calibration(np.mean([m.predict_proba(X, thread_count=-1)[:, 1] for m in models], axis=0).astype(np.float32), man.get("calibration"))
        keep = ["s1_id", "cand_id", "country", pl.col("c_indic").cast(pl.Int8)] + (["label"] if labelled else [])
        out.append(fr.select(keep).with_columns(pl.Series("p_gbm", p)))
    res = pl.concat(out)
    dst = ml_dir(art, split) / (out_name or ("gbm_valid_dense.parquet" if labelled else "gbm_test_dense.parquet"))
    res.write_parquet(dst)
    tmp.unlink(missing_ok=True)
    q = res["p_gbm"]
    log(f"[{split}] CatBoost p on {res.height:,} new pairs -> {dst}: p>=0.5 {int((q >= 0.5).sum()):,}, p>=0.79 {int((q >= 0.79).sum()):,}"
        + (f"; label rate at p>=0.5 {res.filter(pl.col('p_gbm') >= 0.5)['label'].mean() or 0:.3f}, "
           f"at p>=0.79 {res.filter(pl.col('p_gbm') >= 0.79)['label'].mean() or 0:.3f}" if labelled else "") + f" ({time.time() - t0:.0f}s)")
    return dst


def gate(art: Path, scores: str = "llm_scores_valid_merged.parquet") -> dict:
    from .decide import _pred, _sub_blocks
    from .gbm_harness import valid_blocks
    from .llm.metrics import block_f05
    from ..io import load_json
    mdir = ml_dir(art, "train")
    cols = ["s1_id", "cand_id", "country", "label", "p_gbm"]
    v0 = pl.read_parquet(mdir / "gbm_valid.parquet", columns=cols).sort(["s1_id", "cand_id"])
    off0, nt0 = valid_blocks(v0, art)
    ent = v0.select("s1_id", "country").unique("s1_id", maintain_order=True).with_columns(pl.Series("n_truth", nt0))
    d = pl.read_parquet(mdir / "gbm_valid_dense.parquet", columns=cols).join(v0.select("s1_id").unique(), on="s1_id", how="semi")
    v1 = pl.concat([v0, d]).sort(["s1_id", "cand_id"])
    off1 = np.zeros(ent.height + 1, dtype=np.int64)
    np.cumsum(v1.group_by("s1_id", maintain_order=True).len()["len"].to_numpy(), out=off1[1:])
    rules = load_json(mdir / "decision_rule.json")
    rep: dict = {}
    india = (ent["country"] == "India").to_numpy()
    nt = ent["n_truth"].to_numpy()
    for name, v, off in (("before", v0, off0), ("after", v1, off1)):
        y, p = v["label"].to_numpy().astype(np.int64), v["p_gbm"].to_numpy().astype(np.float64)
        f = block_f05(_pred(p, off, rules["gbm_only"]), y, off, nt)
        rep[f"catboost_{name}"] = {"all": float(f.mean()), "India": float(f[india].mean()), "ceiling": float(block_f05(y == 1, y, off, nt).mean())}
        # blend on valid_llm: LLM p on the scored block candidates, CatBoost p elsewhere (new rows included)
        s = pl.read_parquet(mdir / (scores if name == "after" else "llm_scores_valid_merged.parquet"), columns=["s1_id", "cand_id", "p_llm"])
        in_llm = ent["s1_id"].is_in(s["s1_id"].unique().implode()).to_numpy()
        rows, offl = _sub_blocks(off, in_llm)
        vl = v.filter(pl.Series(rows)).with_row_index("_i").join(s, on=["s1_id", "cand_id"], how="left").sort("_i")
        pl_ = vl["p_llm"].cast(pl.Float64).to_numpy()
        w = rules["blend"]["w"]
        pb = np.where(np.isnan(pl_), vl["p_gbm"].cast(pl.Float64).to_numpy(), w * np.nan_to_num(pl_) + (1 - w) * vl["p_gbm"].cast(pl.Float64).to_numpy())
        fl = block_f05(_pred(pb, offl, rules["blend"]), vl["label"].to_numpy().astype(np.int64), offl, nt[in_llm])
        rep[f"blend_valid_llm_{name}"] = {"all": float(fl.mean()), "India": float(fl[india[in_llm]].mean())}
    for k in ("catboost", "blend_valid_llm"):
        rep[f"{k}_delta"] = rep[f"{k}_after"]["all"] - rep[f"{k}_before"]["all"]
    save_json(rep, mdir / "dense_gate.json")
    for k, v in rep.items():
        log(f"gate {k}: {v}")
    return rep


def blocks(art: Path, p_min: float, k_max: int = 16, jobs: int = 1, split: str = "test") -> dict:
    from .blocks import BlocksConfig, Tokenizer, write_subset
    from .blocks_aug import _ranked, candidate_rows, fit_to_tokens, texts_for
    cfg = BlocksConfig(artifacts_dir=art, k_max=k_max)
    tk = Tokenizer(cfg.tokenizer, cfg.k_max)
    labelled = split == "train"
    mdir = ml_dir(art, split)
    gbm_old, gbm_new, scores = (("gbm_valid.parquet", "gbm_valid_dense.parquet", "llm_scores_valid_merged.parquet") if labelled
                                else ("gbm_test.parquet", "gbm_test_dense.parquet", "llm_scores_test.parquet"))
    d = pl.read_parquet(mdir / gbm_new, columns=["s1_id", "cand_id", "p_gbm"]).filter(pl.col("p_gbm") >= p_min)
    flat = (pl.read_parquet(mdir / scores, columns=["s1_id", "cand_id"]).join(d.select("s1_id").unique(), on="s1_id", how="semi")
              .with_columns(pl.lit(True).alias("in_block")))
    d = d.join(flat.select("s1_id").unique(), on="s1_id", how="semi")      # only entities that have an LLM block to extend
    touched = d.select("s1_id").unique()
    old = pl.scan_parquet(mdir / gbm_old).select("s1_id", "cand_id", "p_gbm").join(flat.lazy(), on=["s1_id", "cand_id"], how="inner").collect(engine="streaming")
    rows = pl.concat([old.select("s1_id", "cand_id", pl.col("p_gbm").alias("p_cb"), "in_block"),
                      d.select("s1_id", "cand_id", pl.col("p_gbm").alias("p_cb"), pl.lit(False).alias("in_block"))])
    n0 = rows.height
    rows = (rows.sort(["s1_id", "p_cb"], descending=[False, True]).with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("_r"))
                .filter(pl.col("_r") <= k_max).drop("_r"))
    st = {"p_min": p_min, "entities_touched": touched.height, "new_candidates": d.height, "dropped_for_cap": n0 - rows.height}
    log(f"[{split}] touched {touched.height:,} entities, +{d.height:,} dense candidates (p >= {p_min}), {n0 - rows.height:,} dropped by the cap")
    feat = candidate_rows(art, split, rows.select("s1_id", "cand_id"), labelled)
    sel = _ranked(rows.join(feat, on=["s1_id", "cand_id"], how="inner"))
    e0 = sel.group_by("s1_id").agg(pl.col("country").first())
    s1_text, pool = texts_for(art / split, e0["s1_id"], sel["cand_id"].unique())
    sel, st["dropped_for_tokens"] = fit_to_tokens(cfg, tk, e0, sel, s1_text, pool, labelled=False)
    e = sel.group_by("s1_id").agg(pl.col("country").first(), pl.len().alias("n_sel"), pl.col("p_gbm").max().alias("max_p"),
                                  pl.col("in_block").not_().sum().alias("n_added"))
    if labelled:
        from .gbm_harness import valid_blocks
        v0 = pl.read_parquet(mdir / gbm_old, columns=["s1_id", "cand_id", "label"]).sort(["s1_id", "cand_id"])
        _, nt = valid_blocks(v0, art)
        ent = v0.select("s1_id").unique(maintain_order=True).with_columns(pl.Series("n_truth", nt))
        pos = (pl.concat([v0.join(touched, on="s1_id", how="semi"), pl.read_parquet(mdir / gbm_new, columns=["s1_id", "cand_id", "label"])
                          .join(touched, on="s1_id", how="semi")]).group_by("s1_id").agg(pl.col("label").cast(pl.Int32).sum().alias("n_pos_all")))
        e = (e.join(ent, on="s1_id", how="left").join(pos, on="s1_id", how="left")
              .with_columns((pl.col("n_truth") == 0).alias("singleton"), pl.lit(0, pl.Int16).alias("n_pos_block")))
        note = f"proxy (prompt); membership = scored block + dense candidates with CatBoost p >= {p_min}"
        st["valid_dense"] = write_subset(cfg, tk, "valid_dense", e.sort("s1_id"), sel, s1_text, pool, note)
        save_json(st, mdir / "dense_blocks_stats.json")
        return st
    rank = pl.read_parquet(mdir / "gbm_entity_test.parquet", columns=["s1_id", "route_rank"])
    e = e.join(rank, on="s1_id", how="left").sort("route_rank", nulls_last=True).with_row_index("_i")
    note = f"proxy (prompt); membership = scored block + dense candidates with CatBoost p >= {p_min}"
    for j in range(jobs):
        name = "test_dense" if jobs == 1 else f"test_dense{j}"
        st[name] = write_subset(cfg, tk, name, e.filter(pl.col("_i") % jobs == j).drop("_i"), sel, s1_text, pool, note)
    save_json(st, mdir / "dense_blocks_stats.json")
    return st


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "llm-blocks":
        return llm_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--split", required=True, choices=["train", "test"])
    f.add_argument("--hits", required=True, help="folder holding dense_hits_<split>_r*.parquet (a fetched dense job)")
    f.add_argument("--max-rank", type=int, default=20)
    f.add_argument("--min-cos", type=float, default=0.0)
    f.add_argument("--min-addr", type=float, default=-1.0, help="rapidfuzz token_set_ratio of the addresses; -1 = off")
    f.add_argument("--hits-tags", default="", help="comma list of hit split tags (default: the split name)")
    f.add_argument("--part-name", default=DENSE_PART)
    f.add_argument("--out-name", default="")
    g = sub.add_parser("gate")
    g.add_argument("--scores", default="llm_scores_valid_merged.parquet", help="LLM scores for the 'after' side (train/ml/...)")
    b = sub.add_parser("blocks")
    b.add_argument("--p-min", type=float, default=0.05)
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--split", default="test", choices=["train", "test"])
    for x in (f, g, b):
        x.add_argument("--artifacts-dir", default="artifacts")
    a = p.parse_args(argv)
    art = Path(a.artifacts_dir)
    if a.cmd == "features":
        features(art, a.split, Path(a.hits), max_rank=a.max_rank, min_cos=a.min_cos, min_addr=a.min_addr,
                 hits_tags=tuple(t for t in a.hits_tags.split(",") if t), part_name=a.part_name, out_name=a.out_name)
    elif a.cmd == "gate":
        gate(art, a.scores)
    else:
        blocks(art, a.p_min, jobs=a.jobs, split=a.split)


def new_part(art: Path, split: str, hits_dir: Path, max_rank: int, min_cos: float) -> pl.DataFrame:
    """Write candidate_parts/India__zz_dense.parquet (new dense pairs after the rank/cos filter) and return its rows."""
    tag = "train" if split == "train" else "test"
    hits = pl.concat([pl.read_parquet(f) for f in sorted(Path(hits_dir).rglob(f"dense_hits_{tag}_r*.parquet"))]).unique(["s1_id", "cand_id"])
    india = [f for f in candidate_part_files(art / split) if f.name.startswith("India__") and f.name != DENSE_PART]
    old_keys = pl.concat([pl.scan_parquet(f).select("s1_id", "cand_id") for f in india]).join(
        hits.lazy().select("s1_id").unique(), on="s1_id", how="semi").collect(engine="streaming")
    new = hits.join(old_keys, on=["s1_id", "cand_id"], how="anti").filter((pl.col("rank") <= max_rank) & (pl.col("cos") >= min_cos))
    rows = (new.with_columns(
        pl.col("cand_id").str.slice(0, 2).alias("src"), pl.lit("India").alias("country"), pl.lit("unknown").alias("state_match"),
        pl.lit(1, pl.Int8).alias("n_blockers"), (0.5 / (60 + pl.col("rank"))).cast(pl.Float32).alias("rrf_score"),
        (60 + pl.col("rank")).cast(pl.Int32).alias("rrf_rank"), pl.col("rank").cast(pl.Int32).alias("rank_dense"),
        pl.col("cos").cast(pl.Float32).alias("cos_dense"))
        .with_columns([pl.lit(None, PART_SCHEMA[c]).alias(c) for c in ("rank_exact", "rank_word", "rank_char_name", "rank_char_addr",
                                                                      "cos_exact", "cos_word", "cos_char_name", "cos_char_addr")])
        .select([pl.col(c).cast(t) for c, t in PART_SCHEMA.items()]))
    if split == "train":
        rows = rows.join(_truth(art).with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "cand_id"], how="left").with_columns(
            pl.col("label").fill_null(0).cast(pl.Int8))
    rows.write_parquet(art / split / "candidate_parts" / DENSE_PART)
    log(f"[{split}] {DENSE_PART}: {rows.height:,} new pairs (rank <= {max_rank}, cos >= {min_cos}) for {rows['s1_id'].n_unique():,} entities")
    return rows


def llm_blocks(art: Path, split: str, rows: pl.DataFrame, llm_rank: int, llm_cos: float, jobs: int, k_max: int = 16, prefix: str = "test_dense") -> dict:
    """Blocks for the LLM before CatBoost has seen the new pairs: scored block + new pairs with rank <= llm_rank and
    cos >= llm_cos (placeholder CatBoost p 0.5 so the cap / token fit drops the weakest OLD members first)."""
    from .blocks import BlocksConfig, Tokenizer, write_subset
    from .blocks_aug import _ranked, candidate_rows, fit_to_tokens, texts_for
    cfg = BlocksConfig(artifacts_dir=art, k_max=k_max)
    tk = Tokenizer(cfg.tokenizer, cfg.k_max)
    labelled = split == "train"
    mdir = ml_dir(art, split)
    gbm_old, scores = ("gbm_valid.parquet", "llm_scores_valid_merged_base.parquet") if labelled else ("gbm_test.parquet", "llm_scores_test_base.parquet")
    d = rows.filter((pl.col("rank_dense") <= llm_rank) & (pl.col("cos_dense") >= llm_cos)).select("s1_id", "cand_id", "cos_dense")
    flat = (pl.read_parquet(mdir / scores, columns=["s1_id", "cand_id"]).join(d.select("s1_id").unique(), on="s1_id", how="semi")
              .with_columns(pl.lit(True).alias("in_block")))
    d = d.join(flat.select("s1_id").unique(), on="s1_id", how="semi")
    touched = d.select("s1_id").unique()
    old = pl.scan_parquet(mdir / gbm_old).select("s1_id", "cand_id", "p_gbm").join(flat.lazy(), on=["s1_id", "cand_id"], how="inner").collect(engine="streaming")
    allr = pl.concat([old.select("s1_id", "cand_id", pl.col("p_gbm").cast(pl.Float32).alias("p_cb"), "in_block"),
                      d.select("s1_id", "cand_id", pl.lit(0.5, pl.Float32).alias("p_cb"), pl.lit(False).alias("in_block"))])
    n0 = allr.height
    allr = (allr.sort(["s1_id", "p_cb"], descending=[False, True]).with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("_r"))
                .filter(pl.col("_r") <= k_max).drop("_r"))
    st = {"llm_rank": llm_rank, "llm_cos": llm_cos, "entities": touched.height, "new_candidates": d.height, "dropped_for_cap": n0 - allr.height}
    log(f"[{split}] LLM delta: {touched.height:,} entities, +{d.height:,} dense candidates, {n0 - allr.height:,} old members dropped by the cap")
    feat = candidate_rows(art, split, allr.select("s1_id", "cand_id"), labelled)
    sel = _ranked(allr.join(feat, on=["s1_id", "cand_id"], how="inner"))
    e0 = sel.group_by("s1_id").agg(pl.col("country").first())
    s1_text, pool = texts_for(art / split, e0["s1_id"], sel["cand_id"].unique())
    sel, st["dropped_for_tokens"] = fit_to_tokens(cfg, tk, e0, sel, s1_text, pool, labelled=False)
    e = sel.group_by("s1_id").agg(pl.col("country").first(), pl.len().alias("n_sel"), pl.col("p_gbm").max().alias("max_p"),
                                  pl.col("in_block").not_().sum().alias("n_added"))
    note = f"proxy (prompt); membership = scored block + dense rank <= {llm_rank}, cos >= {llm_cos}"
    if labelled:
        from .gbm_harness import valid_blocks
        v0 = pl.read_parquet(mdir / gbm_old, columns=["s1_id", "cand_id", "label"]).sort(["s1_id", "cand_id"])
        _, nt = valid_blocks(v0, art)
        ent = v0.select("s1_id").unique(maintain_order=True).with_columns(pl.Series("n_truth", nt))
        pos = (pl.concat([v0.join(touched, on="s1_id", how="semi"), rows.select("s1_id", "cand_id", "label").join(touched, on="s1_id", how="semi")])
                 .group_by("s1_id").agg(pl.col("label").cast(pl.Int32).sum().alias("n_pos_all")))
        e = (e.join(ent, on="s1_id", how="left").join(pos, on="s1_id", how="left")
              .with_columns((pl.col("n_truth") == 0).alias("singleton"), pl.lit(0, pl.Int16).alias("n_pos_block")))
        st["valid_dense"] = write_subset(cfg, tk, "valid_dense", e.sort("s1_id"), sel, s1_text, pool, note)
    else:
        best = d.group_by("s1_id").agg(pl.col("cos_dense").max().alias("_c"))
        e = e.join(best, on="s1_id", how="left").sort("_c", descending=True).drop("_c").with_row_index("_i")
        for j in range(jobs):
            st[f"{prefix}{j}"] = write_subset(cfg, tk, f"{prefix}{j}", e.filter(pl.col("_i") % jobs == j).drop("_i"), sel, s1_text, pool, note)
    save_json(st, mdir / "dense_llm_blocks_stats.json")
    return st


def llm_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--hits", required=True)
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--max-rank", type=int, default=20)
    p.add_argument("--min-cos", type=float, default=0.88)
    p.add_argument("--llm-rank", type=int, default=5)
    p.add_argument("--llm-cos", type=float, default=0.90)
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args(argv)
    art = Path(a.artifacts_dir)
    rows = new_part(art, a.split, Path(a.hits), a.max_rank, a.min_cos)
    llm_blocks(art, a.split, rows, a.llm_rank, a.llm_cos, a.jobs)


if __name__ == "__main__":
    main()
