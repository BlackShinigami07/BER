"""Augment the LLM blocks with the strong CatBoost candidates the proxy ranking left out (touched entities only).

    uv run python -m business_entity_resolution.ml blocks-aug --artifacts-dir artifacts --subsets valid,test --p-min 0.02 --test-jobs 2

Why: blocks.py cut the blocks with the proxy p (top-12, up to 16) before CatBoost existed. On valid_llm 2.5% of the true
matches sit below that cut although CatBoost puts them at median rank 3, so the re-ranker never sees them: perfect
decisions inside the current blocks reach F0.5 0.981, inside CatBoost's top-12 0.990 (analysis of 2026-09-27).

What: an entity is touched when a candidate outside its block has CatBoost p >= --p-min. Its new block = old block +
those candidates, capped at --k-max by CatBoost p (the lowest-p members go; CatBoost still decides them outside the
block in export). Prompt lines keep the proxy p and the proxy order, i.e. exactly what the model was trained with; only
the membership changes. Untouched entities keep their old blocks and their old scores.

Outputs: blocks/valid_aug (train split, for `decide`) and blocks/test_aug<j> (test split; entity i in CatBoost's
route_rank order goes to job i % --test-jobs, one folder per scoring job, most uncertain first). Score them with
aws/sagemaker_launch.py score --splits ..., then scripts/ml/merge_test_scores.py replaces the touched entities.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from ..io import load_frame, save_json
from ..progress import log, pbar, stage
from .blocks import BlocksConfig, Tokenizer, build_shard, write_subset
from .common import CAND_COLS, GBM_FILES, attach_p_gbm, blocks_dir, candidate_part_files, load_folds, ml_dir

BASE_SUBSET = {"valid": "valid", "test": "test"}     # augmented subset -> the block set whose membership it extends


def membership(art: Path, base: str, labelled: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(entities with n_truth/singleton/n_pos_block when labelled, flat (s1_id, cand_id, in_block=True)) of the base blocks."""
    cols = ["s1_id", "cand_ids"] + (["n_truth", "singleton", "n_pos_block"] if labelled else [])
    b = pl.read_parquet(blocks_dir(art, base) / "*.parquet", columns=cols)
    flat = (b.select("s1_id", "cand_ids").explode("cand_ids").rename({"cand_ids": "cand_id"})
              .with_columns(pl.lit(True).alias("in_block")))
    log(f"[{base}] {b.height:,} base blocks, {flat.height:,} members")
    return b.drop("cand_ids"), flat


def select_rows(art: Path, split: str, ents: pl.DataFrame, flat: pl.DataFrame, p_min: float, k_max: int) -> tuple[pl.DataFrame, dict]:
    """New block membership (s1_id, cand_id, p_cb, in_block) of the touched entities, from one streaming pass over CatBoost's
    candidate probabilities."""
    files = [ml_dir(art, split) / f for f in GBM_FILES[split] if (ml_dir(art, split) / f).exists()]
    if not files:
        raise FileNotFoundError(f"no CatBoost probabilities {GBM_FILES[split]} under {ml_dir(art, split)} (run `gbm predict`)")
    gbm = pl.concat([pl.scan_parquet(f).select("s1_id", "cand_id", pl.col("p_gbm").cast(pl.Float32).alias("p_cb")) for f in files])
    df = (gbm.join(ents.lazy().select("s1_id"), on="s1_id", how="semi")
             .join(flat.lazy(), on=["s1_id", "cand_id"], how="left")
             .with_columns(pl.col("in_block").fill_null(False))
             .filter(pl.col("in_block") | (pl.col("p_cb") >= p_min))
             .collect(engine="streaming"))
    touched = df.filter(~pl.col("in_block")).select("s1_id").unique()
    df = df.join(touched, on="s1_id", how="semi")
    n_before = df.height
    df = (df.sort(["s1_id", "p_cb"], descending=[False, True])
            .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("_r")).filter(pl.col("_r") <= k_max).drop("_r"))
    st = {"entities": ents.height, "entities_touched": touched.height, "touched_share": round(touched.height / max(1, ents.height), 4),
          "candidates_added": int((~df["in_block"]).sum()), "members_dropped_for_cap": int(n_before - df.height),
          "new_block_size_mean": round(df.height / max(1, touched.height), 2)}
    log(f"[{split}] touched {st['entities_touched']:,} of {ents.height:,} entities ({st['touched_share']:.1%}): "
        f"+{st['candidates_added']:,} candidates (CatBoost p >= {p_min}), {st['members_dropped_for_cap']:,} members dropped by the k_max={k_max} cap")
    return df, st


def candidate_rows(art: Path, split: str, keys: pl.DataFrame, labelled: bool) -> pl.DataFrame:
    """Prompt fields (src, n_blockers, rrf_rank, country, label) and the proxy p for the given (s1_id, cand_id) keys."""
    folds = load_folds(art) if labelled else None
    cols = CAND_COLS + (["label"] if labelled else [])
    keep = ["s1_id", "cand_id", "src", "p_gbm", "n_blockers", "rrf_rank", "country"] + (["label"] if labelled else [])
    out = []
    for f in pbar(candidate_part_files(art / split), desc=f"features {split}", unit="part"):
        lf = pl.scan_parquet(f).select(cols).join(keys.lazy().select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi")
        lf, _ = attach_p_gbm(lf, art, split, folds, source="proxy")
        out.append(lf.select(keep).collect())
    feat = pl.concat(out)
    if feat.height != keys.height:
        raise RuntimeError(f"{keys.height - feat.height:,} selected candidates have no candidate-part row (gbm files and candidate parts differ?)")
    return feat


def texts_for(split_dir: Path, s1_ids: pl.Series, cand_ids: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """blocks._texts restricted to the needed ids, one source at a time (the full pools do not fit next to the candidates)."""
    clean = lambda c: pl.col(c).fill_null("").str.replace_all(r"[\r\n\t]+", " ").str.strip_chars()

    def load(i: int, id_col: str, ids: pl.Series) -> pl.DataFrame:
        return (load_frame(split_dir / f"normalized_source{i}.pkl").filter(pl.col("entity_id").is_in(ids.implode()))
                .select(pl.col("entity_id").alias(id_col), clean("business_name").alias("name"), clean("business_address").alias("addr"),
                        clean("name_core").alias("core"), pl.col("name_script").fill_null("latin").alias("script")))

    s1 = load(1, "s1_id", s1_ids)
    pool = pl.concat([load(i, "cand_id", cand_ids) for i in (2, 3)])
    return s1, pool


def _ranked(sel: pl.DataFrame) -> pl.DataFrame:
    """Prompt order = the proxy order the model was trained with."""
    return (sel.sort(["s1_id", "p_gbm", "rrf_rank"], descending=[False, True, False])
               .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("sel_rank")))


def fit_to_tokens(cfg: BlocksConfig, tk: Tokenizer, e: pl.DataFrame, sel: pl.DataFrame, s1_text: pl.DataFrame, pool: pl.DataFrame,
                  labelled: bool, max_rounds: int = 4) -> tuple[pl.DataFrame, int]:
    """build_shard keeps the first lines of a block that exceeds max_tokens, and the added candidates (low proxy p) sit last:
    on valid the cap cut 13% of them. Tokenise, and for every over-long block drop its lowest-CatBoost-p members instead."""
    n_dropped = 0
    for rnd in range(max_rounds):
        kept = []
        for i in range(0, e.height, cfg.shard_blocks):
            t, _ = build_shard(cfg, tk, e.slice(i, cfg.shard_blocks), sel, s1_text, pool, labelled, {"capped_blocks": 0, "capped_cands": 0, "capped_pos": 0})
            kept.append(pl.from_arrow(t.select(["s1_id", "n_cands"])))
        k = pl.concat(kept).with_columns(pl.col("n_cands").cast(pl.Int64))
        over = (k.join(sel.group_by("s1_id").len(), on="s1_id").with_columns((pl.col("len") - pl.col("n_cands")).alias("deficit"))
                  .filter(pl.col("deficit") > 0))
        if over.height == 0:
            break
        low = (sel.join(over.select("s1_id", "deficit"), on="s1_id", how="inner").sort(["s1_id", "p_cb"])
                  .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("_low")))
        drop = low.filter(pl.col("_low") <= pl.col("deficit")).select("s1_id", "cand_id")
        n_dropped += drop.height
        sel = _ranked(sel.join(drop, on=["s1_id", "cand_id"], how="anti"))
        log(f"fit round {rnd + 1}: {over.height:,} blocks over {cfg.max_tokens} tokens -> dropped {drop.height:,} lowest-CatBoost-p members")
    return sel, n_dropped


def augment(cfg: BlocksConfig, tk: Tokenizer, subset: str, p_min: float, test_jobs: int) -> dict:
    art, base = cfg.artifacts_dir, BASE_SUBSET[subset]
    split = "train" if subset == "valid" else "test"
    labelled = split == "train"
    ents, flat = membership(art, base, labelled)
    rows, st = select_rows(art, split, ents, flat, p_min, cfg.k_max)
    del flat
    if rows.height == 0:
        log(f"[{subset}] nothing to augment")
        return st
    feat = candidate_rows(art, split, rows.select("s1_id", "cand_id"), labelled)
    sel = _ranked(rows.join(feat, on=["s1_id", "cand_id"], how="inner"))
    e0 = sel.group_by("s1_id").agg(pl.col("country").first())
    s1_text, pool = texts_for(art / split, e0["s1_id"], sel["cand_id"].unique())
    sel, st["members_dropped_for_tokens"] = fit_to_tokens(cfg, tk, e0, sel, s1_text, pool, labelled=False)   # only n_cands is needed
    agg = [pl.col("country").first(), pl.len().alias("n_sel"), pl.col("p_gbm").max().alias("max_p"), pl.col("in_block").not_().sum().alias("n_added")]
    e = sel.group_by("s1_id").agg(agg)
    st["candidates_added"] = int(e["n_added"].sum())
    if labelled:
        # n_pos_all (every retrieved positive) for write_subset's coverage stats, from CatBoost's labelled valid rows
        pos = (pl.scan_parquet(ml_dir(art, split) / "gbm_valid.parquet").select("s1_id", "label")
                 .join(e.lazy().select("s1_id"), on="s1_id", how="semi").group_by("s1_id").agg(pl.col("label").cast(pl.Int32).sum().alias("n_pos_all")).collect())
        e = e.join(pos, on="s1_id", how="left").join(ents.select("s1_id", "n_truth", "singleton", "n_pos_block"), on="s1_id", how="left")
        new_pos = sel.group_by("s1_id").agg(pl.col("label").cast(pl.Int32).sum().alias("n_pos_new"))
        e = e.join(new_pos, on="s1_id", how="left")
        tot = int(ents["n_truth"].sum())
        old_cov = int(ents["n_pos_block"].sum())
        new_cov = old_cov - int(e["n_pos_block"].sum()) + int(e["n_pos_new"].sum())
        st.update(truth_in_blocks_before=round(old_cov / tot, 4), truth_in_blocks_after=round(new_cov / tot, 4),
                  positives_lost_to_cap=int((e["n_pos_block"] > e["n_pos_new"]).sum()))
        log(f"[{subset}] true matches inside the blocks: {old_cov / tot:.2%} -> {new_cov / tot:.2%} of all truth "
            f"(entities whose block lost a positive to the cap: {st['positives_lost_to_cap']:,})")
        e = e.drop("n_pos_new")
    src_note = f"proxy (prompt); membership = {base} blocks + CatBoost p >= {p_min}, capped at k_max by CatBoost p"
    if not labelled:
        rank = pl.read_parquet(ml_dir(art, split) / "gbm_entity_test.parquet", columns=["s1_id", "route_rank"])
        e = e.join(rank, on="s1_id", how="left").sort("route_rank", nulls_last=True).with_row_index("_i")
        for j in range(test_jobs):
            ej = e.filter(pl.col("_i") % test_jobs == j).drop("_i")
            st[f"{subset}_aug{j}"] = write_subset(cfg, tk, f"{subset}_aug{j}", ej, sel, s1_text, pool, src_note)
    else:
        st[f"{subset}_aug"] = write_subset(cfg, tk, f"{subset}_aug", e.sort("s1_id"), sel, s1_text, pool, src_note)
    return st


def run(cfg: BlocksConfig, subsets: tuple[str, ...], p_min: float, test_jobs: int) -> dict:
    t0 = time.time()
    tk = Tokenizer(cfg.tokenizer, cfg.k_max)
    report = {"p_min": p_min, "k_max": cfg.k_max, "test_jobs": test_jobs}
    for s in subsets:
        with stage(f"blocks-aug {s}"):
            report[s] = augment(cfg, tk, s, p_min, test_jobs)
            save_json(report, ml_dir(cfg.artifacts_dir, "train" if s == "valid" else "test") / "blocks_aug_stats.json")
    log(f"blocks-aug done in {time.time() - t0:.0f}s")
    return report


def main(argv: list[str] | None = None) -> None:
    d = BlocksConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", default=str(d.artifacts_dir))
    p.add_argument("--subsets", default="valid,test", help="valid (-> blocks/valid_aug) and/or test (-> blocks/test_aug<j>)")
    p.add_argument("--p-min", type=float, default=0.02, help="CatBoost p above which an outside candidate joins the block")
    p.add_argument("--k-max", type=int, default=d.k_max, help="block size cap (the model trained on blocks of <= 16)")
    p.add_argument("--test-jobs", type=int, default=2, help="number of test_aug folders (one scoring job each)")
    p.add_argument("--tokenizer", default=d.tokenizer)
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--shard-blocks", type=int, default=d.shard_blocks)
    a = p.parse_args(argv)
    cfg = BlocksConfig(artifacts_dir=Path(a.artifacts_dir), tokenizer=a.tokenizer, k_max=a.k_max, max_tokens=a.max_tokens,
                       shard_blocks=a.shard_blocks)
    run(cfg, tuple(s for s in a.subsets.split(",") if s), a.p_min, a.test_jobs)


if __name__ == "__main__":
    main()
