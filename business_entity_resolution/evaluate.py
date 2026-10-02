"""Stage `evaluate` (train): blocking quality against the ground truth (plan D8), streaming over candidate parts."""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import BLOCKER_NAMES, BlockingConfig
from .io import (gt_to_pairs, list_countries, load_frame, load_json, load_pkl, normalized_part, read_ground_truth, save_json,
                 save_pkl, scan_candidates)
from .progress import log


def f05_ceiling(recalls: np.ndarray) -> float:
    r = recalls
    return float(np.mean(np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)))


def stage_evaluate(cfg: BlockingConfig) -> dict:
    d = cfg.split_dir("train")
    countries = list_countries(d)
    cands = scan_candidates(d)
    s1 = pl.concat([pl.read_parquet(normalized_part(d, "source1", c), columns=["entity_id", "country"]) for c in countries])
    pool_meta = pl.concat([pl.read_parquet(normalized_part(d, s, c), columns=["entity_id", "name_script", "addr_comps"])
                           for c in countries for s in ("source2", "source3") if normalized_part(d, s, c).exists()])
    pool_meta = pool_meta.with_columns((pl.col("addr_comps") == "").alias("addr_empty")).drop("addr_comps")
    pairs = gt_to_pairs(read_ground_truth(cfg.data_dir / "train" / "train_ground_truth.tsv"))
    splits = load_frame(d / "splits.pkl")
    sample_path = d / "sample_ids.pkl"
    if sample_path.exists():
        ids = load_pkl(sample_path)
        s1 = s1.filter(pl.col("entity_id").is_in(ids["s1_ids"]))
        pairs = pairs.filter(pl.col("s1_id").is_in(ids["s1_ids"]) & pl.col("cand_id").is_in(ids["pool_ids"]))
        log(f"evaluation restricted to the sampled universe: {s1.height:,} S1, {pairs.height:,} GT pairs")

    # ---- pair-level: which GT pairs were retrieved, by which blockers (streaming join against the parts)
    sel = cands.select("s1_id", "cand_id", "rrf_rank", "n_blockers", *[f"rank_{b}" for b in BLOCKER_NAMES])
    got = pairs.lazy().join(sel, on=["s1_id", "cand_id"], how="left").collect(engine="streaming")
    got = got.with_columns(pl.col("rrf_rank").is_not_null().alias("hit"))
    got = got.join(s1.rename({"entity_id": "s1_id"}), on="s1_id").join(pool_meta.rename({"entity_id": "cand_id"}), on="cand_id", how="left") \
             .join(splits.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    rep: dict = {}
    rep["pair_recall"] = float(got["hit"].mean())
    rep["pair_recall_by_country"] = {k: float(v) for k, v in got.group_by("country").agg(pl.col("hit").mean()).iter_rows()}
    rep["pair_recall_by_script"] = {k: float(v) for k, v in got.group_by("name_script").agg(pl.col("hit").mean()).iter_rows()}
    ae = got.filter(pl.col("addr_empty"))
    rep["pair_recall_addr_empty"] = float(ae["hit"].mean()) if ae.height else None
    rep["pair_recall_by_fold"] = {k: float(v) for k, v in got.group_by("fold").agg(pl.col("hit").mean()).iter_rows()}
    per_blocker = {}
    for b in BLOCKER_NAMES:
        col = f"rank_{b}"
        if got[col].null_count() == got.height:
            continue
        only_b = got.filter(pl.col("hit") & (pl.col("n_blockers") == 1) & pl.col(col).is_not_null()).height
        without = got.filter(pl.col("hit") & ~((pl.col("n_blockers") == 1) & pl.col(col).is_not_null())).height
        per_blocker[b] = {"pairs_found": int(got.filter(pl.col("hit") & pl.col(col).is_not_null()).height),
                          "unique_pairs": int(only_b), "recall_without": float(without / got.height)}
    rep["per_blocker"] = per_blocker
    rep["pair_recall_at_cap"] = {int(c): float((got["rrf_rank"].fill_null(10**9) <= c).mean()) for c in cfg.report_caps if c <= cfg.cap}

    # ---- entity-level
    ent = got.group_by("s1_id").agg(pl.col("hit").mean().alias("recall"), pl.len().alias("n_true"), pl.col("country").first(), pl.col("fold").first())
    all_s1 = s1.join(ent, left_on="entity_id", right_on="s1_id", how="left")
    singles = all_s1.filter(pl.col("n_true").is_null()); matched = all_s1.filter(pl.col("n_true").is_not_null())
    rep["n_s1"] = int(all_s1.height); rep["n_singletons"] = int(singles.height)
    rep["entity_recall_mean"] = float(matched["recall"].mean()) if matched.height else None
    rep["entity_full_recall_rate"] = float((matched["recall"] == 1.0).mean()) if matched.height else None
    rep["entity_recall_by_country"] = {k: float(v) for k, v in matched.group_by("country").agg(pl.col("recall").mean()).iter_rows()}
    rec = matched["recall"].to_numpy() if matched.height else np.zeros(0)
    rep["f05_ceiling_matched_only"] = f05_ceiling(rec) if matched.height else None
    rep["f05_ceiling_incl_singletons"] = float((f05_ceiling(rec) * matched.height + 1.0 * singles.height) / max(all_s1.height, 1)) if matched.height else 1.0
    # ---- candidates per S1 (streaming count), singleton load
    per_s1 = cands.group_by("s1_id").agg(pl.len().alias("n")).collect(engine="streaming")
    cnt = all_s1.join(per_s1, left_on="entity_id", right_on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
    n = cnt["n"].to_numpy()
    rep["cands_per_s1"] = {"mean": float(n.mean()), "median": float(np.median(n)), "p95": float(np.percentile(n, 95)), "max": int(n.max()), "zero": int((n == 0).sum())}
    rep["singleton_cands_mean"] = float(cnt.filter(pl.col("n_true").is_null())["n"].mean()) if singles.height else None
    summ = load_json(d / "block_summary.json")
    rep["n_pool"] = summ.get("n_pool"); rep["n_pairs"] = summ.get("n_pairs")
    rep["reduction_ratio"] = 1.0 - summ["n_pairs"] / max(summ["n_s1"] * summ["n_pool"], 1)
    rep["config"] = cfg.to_dict()
    save_json(rep, d / "blocking_metrics.json")
    missed = got.filter(~pl.col("hit")).select("s1_id", "cand_id", "country", "name_script", "addr_empty", "fold")
    save_pkl(missed, d / "missed_pairs.pkl")
    print_report(rep)
    return rep


def print_report(rep: dict) -> None:
    lines = ["", "=" * 78, "BLOCKING EVALUATION (train ground truth)", "=" * 78]
    lines.append(f"S1 entities {rep['n_s1']:,} (singletons {rep['n_singletons']:,}) | candidate pairs {rep.get('n_pairs')} | pool {rep.get('n_pool')}")
    lines.append(f"pair recall        : {rep['pair_recall']:.4f}   by country {fmt(rep['pair_recall_by_country'])}")
    lines.append(f"  by name script   : {fmt(rep['pair_recall_by_script'])}   addr-empty: {rep['pair_recall_addr_empty']}")
    lines.append(f"  by fold          : {fmt(rep['pair_recall_by_fold'])}")
    lines.append(f"entity recall mean : {rep['entity_recall_mean']}   full-recall rate {rep['entity_full_recall_rate']}   by country {fmt(rep['entity_recall_by_country'])}")
    lines.append(f"F0.5 ceiling       : {rep['f05_ceiling_incl_singletons']:.4f} (incl. singletons) / {rep['f05_ceiling_matched_only']} (matched only)")
    lines.append(f"cands per S1       : {rep['cands_per_s1']}   singleton cands mean {rep['singleton_cands_mean']}")
    lines.append(f"reduction ratio    : {rep.get('reduction_ratio')}")
    lines.append(f"recall at cap      : {fmt(rep['pair_recall_at_cap'])}")
    lines.append("per blocker        : " + "; ".join(f"{b}: found {v['pairs_found']:,}, unique {v['unique_pairs']:,}, recall w/o {v['recall_without']:.4f}" for b, v in rep["per_blocker"].items()))
    lines.append("=" * 78)
    log("\n".join(lines))


def fmt(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in d.items()) + "}"
