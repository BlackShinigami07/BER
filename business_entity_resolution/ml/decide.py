"""WP-E / stage C: tune the per-entity decision rules and the LLM routing share; report stage A vs A+B
(GEMMA_PLAN.md section 9, CATBOOST_PLAN.md sections 4, 5 and WP-5).

    uv run python -m business_entity_resolution.ml decide --artifacts-dir artifacts [--scores artifacts/train/ml/llm_scores_valid.parquet]

Rule: keep candidates with p >= t, at most the top-m of the entity, and only those >= r x the entity's best p; empty
set when none passes. Exact per-entity F0.5 (n_truth counts every true match, including those blocking missed).

With CatBoost predictions (train/ml/gbm_valid.parquet, from `gbm predict --split valid`) - the normal path:
* gbm_only rule: CatBoost p on every candidate, tuned on ALL of ml_valid (gbm_rule.json);
* blend rule (w, t, m, r), tuned on valid_llm: p = w * p_llm + (1 - w) * p_catboost for the candidates in the entity's
  LLM block, p_catboost for its other candidates (the blocks were cut with the proxy p, so CatBoost may rate a
  candidate outside the block highly);
* routing curve: F0.5 on valid_llm of "blend rule on the top-K% entities by CatBoost's u, gbm_only rule elsewhere"
  for K = 0..50% (and 100%), and K* = the smallest K after which routing more entities adds < 0.0003 F0.5.
Without them it falls back to the blocks' p_gbm (the proxy) with a warning (the pre-CatBoost behaviour).
Writes train/ml/decision_rule.json and train/ml/decide_report.json.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..io import load_json, save_json
from ..progress import log
from .common import blocks_dir, ml_dir
from .llm.metrics import apply_rule, block_f05

W_GRID = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
T_GRID = tuple(np.round(np.arange(0.20, 0.981, 0.01), 2))
M_GRID = (0, 1, 2, 3, 5, 8)
R_GRID = (0.0, 0.5, 0.8)
K_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0)
FLAT = 0.0003     # routing more entities than K* adds less than this macro F0.5 on valid_llm


def load_valid(artifacts_dir, scores: Path | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(blocks, flat candidates) of valid_llm with p_llm joined; flat rows stay grouped by block in block order."""
    cols = ["s1_id", "country", "cand_ids", "labels", "p_gbm", "n_truth", "has_indic", "is_hard", "routed", "singleton"]
    b = pl.read_parquet(blocks_dir(artifacts_dir, "valid") / "*.parquet", columns=cols).with_row_index("blk")
    flat = (b.select("blk", "s1_id", "cand_ids", "labels", "p_gbm").explode(["cand_ids", "labels", "p_gbm"])
              .rename({"cand_ids": "cand_id", "labels": "label"}).with_row_index("pos"))
    if scores is not None and Path(scores).exists():
        s = pl.read_parquet(scores)
        keep = ["s1_id", "cand_id", "p_llm"] + [c for c in ("p_fwd", "p_rev") if c in s.columns]
        flat = flat.join(s.select(keep), on=["s1_id", "cand_id"], how="left").sort("pos")
    else:
        log(f"WARNING: no LLM scores at {scores}: only the stage-A rule can be tuned")
        flat = flat.with_columns(pl.lit(None, dtype=pl.Float32).alias("p_llm"))
    return b, flat


def search(p: np.ndarray, labels: np.ndarray, off: np.ndarray, n_truth: np.ndarray, *, t_grid=T_GRID, m_grid=M_GRID,
           r_grid=R_GRID) -> tuple[dict, float]:
    best, best_f = {}, -1.0
    for m in m_grid:
        for r in r_grid:
            base = apply_rule(p, off, 0.0, m or None, r)
            for t in t_grid:
                f = float(block_f05(base & (p >= t), labels, off, n_truth).mean())
                if f > best_f:
                    best, best_f = {"t": float(t), "m": int(m), "r": float(r)}, f
    return best, best_f


def slice_report(pred: np.ndarray, labels, off, n_truth, b: pl.DataFrame) -> dict:
    f = block_f05(pred, labels, off, n_truth)
    npred = np.add.reduceat(pred.astype(np.int64), off[:-1])
    out = {"f05": float(f.mean()), "n": int(len(f))}
    for c in b["country"].unique().sort().to_list():
        m = (b["country"] == c).to_numpy()
        out[f"f05_{c}"] = float(f[m].mean())
    for c in ("has_indic", "is_hard", "routed"):
        m = b[c].to_numpy().astype(bool)
        if m.any():
            out[f"f05_{c}"] = float(f[m].mean())
            out[f"f05_not_{c}"] = float(f[~m].mean()) if (~m).any() else None
    single = (np.asarray(n_truth) == 0)
    if single.any():
        out["singleton_accuracy"] = float((npred[single] == 0).mean())
        out["singleton_share"] = float(single.mean())
    return out


def run(artifacts_dir, scores: Path | None) -> dict:
    if (ml_dir(artifacts_dir, "train") / "gbm_valid.parquet").exists():
        return run_catboost(Path(artifacts_dir), scores)
    log("WARNING: no CatBoost predictions (train/ml/gbm_valid.parquet): falling back to the blocks' proxy p_gbm "
        "(run `gbm predict --split valid` for the CatBoost stage A)")
    return run_blocks(artifacts_dir, scores)


def _sub_blocks(off: np.ndarray, keep_ent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row mask, new offsets) keeping the entities flagged in keep_ent."""
    n = np.diff(off)
    rows = np.repeat(keep_ent, n)
    new = np.zeros(int(keep_ent.sum()) + 1, dtype=np.int64)
    np.cumsum(n[keep_ent], out=new[1:])
    return rows, new


def run_catboost(art: Path, scores: Path | None) -> dict:
    from .gbm import _slices, valid_arrays
    t0 = time.time()
    a = valid_arrays(art)
    y, off, n_truth, p_cat, ent = a["y"], a["off"], a["n_truth"], a["p_gbm"], a["ent"]
    rule_f = ml_dir(art, "train") / "gbm_rule.json"
    if rule_f.exists():
        rule_a = load_json(rule_f)["rule"]
    else:
        rule_a, _ = search(p_cat, y, off, n_truth)
    pred_a = _pred(p_cat, off, rule_a)
    p_max = np.maximum.reduceat(p_cat, off[:-1])
    report: dict = {"p_source": "catboost", "n_ml_valid_entities": int(len(off) - 1), "scores": str(scores),
                    "stage_a_ml_valid": {"rule": {"w": 0.0, **rule_a}, **_slices(pred_a, y, off, n_truth, ent, p_max)}}
    log(f"stage A (CatBoost) on all of ml_valid: F0.5 {report['stage_a_ml_valid']['f05']:.4f} with {rule_a}")
    rules = {"gbm_only": {"w": 0.0, **rule_a}, "blend": {"w": 0.0, **rule_a}, "p_source": "catboost"}
    # valid_llm = the entities of the valid blocks (a country-stratified random 30k of ml_valid)
    bdir = blocks_dir(art, "valid")
    have_llm = scores is not None and Path(scores).exists() and bdir.exists()
    if bdir.exists():
        vb = pl.read_parquet(bdir / "*.parquet", columns=["s1_id"])
        in_llm = ent["s1_id"].is_in(vb["s1_id"].implode()).to_numpy()
        rows, off_l = _sub_blocks(off, in_llm)
        y_l, nt_l, pc_l = y[rows], n_truth[in_llm], p_cat[rows]
        ent_l = ent.filter(pl.Series(in_llm))
        pred_al = _pred(pc_l, off_l, rule_a)
        report["stage_a"] = {"rule": {"w": 0.0, **rule_a}, **_slices(pred_al, y_l, off_l, nt_l, ent_l, p_max[in_llm])}
        log(f"stage A (CatBoost) on valid_llm ({int(in_llm.sum()):,} entities): F0.5 {report['stage_a']['f05']:.4f}")
    if not have_llm:
        log(f"no LLM scores at {scores} (or no valid blocks): only the stage-A rule; routing share not tuned")
    else:
        s = pl.read_parquet(scores)
        keep = ["s1_id", "cand_id", "p_llm"] + [c for c in ("p_fwd", "p_rev") if c in s.columns]
        v = a["v"].select("s1_id", "cand_id").filter(pl.Series(rows)).with_row_index("_i")
        j = v.join(s.select(keep), on=["s1_id", "cand_id"], how="left").sort("_i")
        p_llm = j["p_llm"].cast(pl.Float64).to_numpy()
        have = ~np.isnan(p_llm)
        log(f"valid_llm: {have.sum():,} of {len(have):,} candidates carry p_llm (the block); CatBoost p elsewhere")
        best, best_f = {}, -1.0
        for w in W_GRID:
            q = np.where(have, w * np.nan_to_num(p_llm) + (1 - w) * pc_l, pc_l)
            r, f = search(q, y_l, off_l, nt_l)
            report.setdefault("by_w", {})[str(w)] = {"f05": f, **r}
            if f > best_f:
                best, best_f = {"w": float(w), **r}, f
        rules["blend"] = best
        p_bl = np.where(have, best["w"] * np.nan_to_num(p_llm) + (1 - best["w"]) * pc_l, pc_l)
        pred_bl = _pred(p_bl, off_l, best)
        report["stage_ab"] = {"rule": best, **_slices(pred_bl, y_l, off_l, nt_l, ent_l, p_max[in_llm])}
        log(f"stage A+B on valid_llm: F0.5 {best_f:.4f} with {best} (gain {best_f - report['stage_a']['f05']:+.4f})")
        # routing: LLM (blend rule) on the top-K% of valid_llm by CatBoost's u, CatBoost (gbm_only rule) elsewhere
        f_a = block_f05(pred_al, y_l, off_l, nt_l)
        f_b = block_f05(pred_bl, y_l, off_l, nt_l)
        seg = np.repeat(np.arange(len(off_l) - 1), np.diff(off_l))
        u = np.bincount(seg, weights=np.minimum(pc_l, 1 - pc_l))
        order = np.argsort(-u, kind="stable")
        curve = []
        for k in K_GRID:
            routed = np.zeros(len(u), dtype=bool)
            routed[order[:int(round(k * len(u)))]] = True
            curve.append({"top_k_share": k, "routed": int(routed.sum()), "f05": float(np.where(routed, f_b, f_a).mean())})
        fs = np.array([c["f05"] for c in curve if c["top_k_share"] <= 0.5])
        ks = [c["top_k_share"] for c in curve if c["top_k_share"] <= 0.5]
        k_star = next(k for i, k in enumerate(ks) if fs[i:].max() - fs[i] < FLAT)
        rules["route"] = {"top_k_share": k_star, "flat_tolerance": FLAT, "order": "u = sum min(p, 1-p) (CatBoost), descending"}
        report["routing_curve"] = curve
        report["route"] = rules["route"]
        log("routing curve on valid_llm (share routed to the LLM by u -> macro F0.5): "
            + ", ".join(f"{c['top_k_share']:.0%}: {c['f05']:.4f}" for c in curve) + f" -> K* = {k_star:.0%}")
        if "p_rev" in j.columns:   # S9 order sensitivity of the LLM alone (w = 1) on its blocks, CatBoost elsewhere
            variants = {"forward": j["p_fwd"], "reversed": j["p_rev"], "averaged": j["p_llm"]}
            report["order_sensitivity"] = {k: search(np.where(have, np.nan_to_num(q.cast(pl.Float64).to_numpy()), pc_l), y_l, off_l, nt_l)[1]
                                           for k, q in variants.items()}
            log(f"order sensitivity (S9): {report['order_sensitivity']}")
    out = ml_dir(art, "train")
    save_json({**rules, "f05_ml_valid_stage_a": report["stage_a_ml_valid"]["f05"],
               "f05_valid_llm": {"stage_a": report.get("stage_a", {}).get("f05"), "stage_ab": report.get("stage_ab", {}).get("f05")},
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out / "decision_rule.json")
    save_json(report, out / "decide_report.json")
    if "stage_a" in report:
        _print_table(report)
    log(f"decide done in {time.time() - t0:.0f}s -> {out / 'decision_rule.json'}")
    return report


def run_blocks(artifacts_dir, scores: Path | None) -> dict:
    """Pre-CatBoost stage C on the valid_llm blocks' own candidates and their (proxy) p_gbm."""
    t0 = time.time()
    b, flat = load_valid(artifacts_dir, scores)
    off = np.zeros(b.height + 1, dtype=np.int64)
    np.cumsum(b["cand_ids"].list.len().to_numpy(), out=off[1:])
    labels = flat["label"].to_numpy().astype(np.int64)
    n_truth = b["n_truth"].to_numpy()
    p_gbm = flat["p_gbm"].to_numpy().astype(np.float64)
    p_llm = flat["p_llm"].to_numpy()
    have_llm = p_llm is not None and not np.isnan(np.asarray(p_llm, dtype=np.float64)).all()
    report: dict = {"n_blocks": int(b.height), "n_cands": int(len(labels)), "scores": str(scores)}
    rule_a, f_a = search(p_gbm, labels, off, n_truth)
    report["stage_a"] = {"rule": {"w": 0.0, **rule_a}, **slice_report(_pred(p_gbm, off, rule_a), labels, off, n_truth, b)}
    log(f"stage A only: F0.5 {f_a:.4f} with {rule_a}")
    rules = {"gbm_only": {"w": 0.0, **rule_a}, "blend": {"w": 0.0, **rule_a}}
    if have_llm:
        p_llm = np.asarray(p_llm, dtype=np.float64)
        miss = np.isnan(p_llm)
        if miss.any():
            log(f"WARNING: {miss.sum():,} valid candidates have no p_llm (block not scored): p_gbm used for them")
            p_llm = np.where(miss, p_gbm, p_llm)
        best, best_f = {}, -1.0
        for w in W_GRID:
            r, f = search(w * p_llm + (1 - w) * p_gbm, labels, off, n_truth)
            report.setdefault("by_w", {})[str(w)] = {"f05": f, **r}
            if f > best_f:
                best, best_f = {"w": float(w), **r}, f
        p = best["w"] * p_llm + (1 - best["w"]) * p_gbm
        rules["blend"] = best
        report["stage_ab"] = {"rule": best, **slice_report(_pred(p, off, best), labels, off, n_truth, b)}
        log(f"stage A+B: F0.5 {best_f:.4f} with {best} (gain {best_f - f_a:+.4f})")
        if "p_rev" in flat.columns:   # S9 order sensitivity of the LLM alone (w = 1): forward vs reversed vs averaged
            fwd, rev = flat["p_fwd"].to_numpy().astype(np.float64), flat["p_rev"].to_numpy().astype(np.float64)
            report["order_sensitivity"] = {k: search(q, labels, off, n_truth)[1]
                                           for k, q in (("forward", fwd), ("reversed", rev), ("averaged", p_llm))}
            log(f"order sensitivity (S9): {report['order_sensitivity']}")
    out = ml_dir(artifacts_dir, "train")
    save_json({**rules, "f05_valid_llm": {"stage_a": f_a, "stage_ab": report.get("stage_ab", {}).get("f05")},
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out / "decision_rule.json")
    save_json(report, out / "decide_report.json")
    _print_table(report)
    log(f"decide done in {time.time() - t0:.0f}s -> {out / 'decision_rule.json'}")
    return report


def _pred(p, off, rule: dict) -> np.ndarray:
    return apply_rule(p, off, rule["t"], rule.get("m") or None, rule.get("r", 0.0))


def _print_table(report: dict) -> None:
    rows = [k for k in report.get("stage_ab", report["stage_a"]) if k.startswith("f05") or k.startswith("singleton")]
    lines = [f"{'metric (valid_llm)':<28}{'stage A':>10}{'A+B':>10}"]
    for k in rows:
        a = report["stage_a"].get(k)
        ab = report.get("stage_ab", {}).get(k)
        fmt = lambda v: f"{v:>10.4f}" if isinstance(v, float) else f"{'-':>10}"
        lines.append(f"{k:<28}{fmt(a)}{fmt(ab)}")
    log("\n" + "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--scores", default=None, help="default: <artifacts>/train/ml/llm_scores_valid.parquet")
    a = p.parse_args(argv)
    run(Path(a.artifacts_dir), Path(a.scores) if a.scores else ml_dir(a.artifacts_dir, "train") / "llm_scores_valid.parquet")


if __name__ == "__main__":
    main()
