"""WP-E export: apply the tuned decision rules to the test candidates and write output/matching_results.tsv.

    uv run python -m business_entity_resolution.ml export --artifacts-dir artifacts --test-dir dataset/test
    uv run python -m business_entity_resolution.ml export ... --variant llm_only                    # LLM alone (w = 1) on scored entities
    uv run python -m business_entity_resolution.ml export ... --country-variant France=llm_only    # per-country override
    uv run python -m business_entity_resolution.ml export --artifacts-dir dev/mini_artifacts --test-dir dev/mini_dataset/test \
        --output-dir dev/output

Stage-A probabilities are CatBoost's (test/ml/gbm_test.parquet, CATBOOST_PLAN.md) for every candidate. Entities scored
by the LLM (test/ml/llm_scores_test.parquet) use a blend rule p = w * p_llm + (1 - w) * p_catboost on their block
candidates and p_catboost on their other candidates: `blend` (decision_rule.json, w = 0.8) or `llm_only` (w = 1, the
rule tuned at w = 1.0 in decide_report.json), chosen globally with --variant and per country with --country-variant.
Every other entity uses the CatBoost (gbm_only) rule. Without LLM scores this is the CatBoost-only safety submission.
Without gbm_test.parquet it falls back to the proxy p (warning).

Global post-processing (2026-09-27, both validated on labelled valid):
* conflicts (--resolve-conflicts, default on): the truth never assigns one S2/S3 record to two S1 entities, so when
  several entities pick the same record only the highest-p claim is kept;
* uniqueness (--uniqueness, default on when test/ml/uniq_flags.parquet exists; `uniq-flags` builds it): an empty-address
  record with name similarity >= 0.8 is added when exactly one S1 entity in the split has such a name and p >= 0.3, and
  dropped when three or more do (label rates 97.6% / <= 4%).
One row per test S1 entity, ids only from that entity's candidate list, then utils/validate_submission.py is run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

from ..io import load_frame, load_json
from ..progress import log, pbar
from .common import CAND_COLS, attach_p_gbm, candidate_part_files, ml_dir

BUCKET_ROWS = 12_000_000   # CatBoost test rows processed per pass (entity-complete buckets by id hash)
VARIANTS = ("blend", "llm_only")
UNIQ_ADD_P = 0.3           # uniqueness rule: add a unique empty-address record when its p is at least this
UNIQ_DROP_N = 3            # ... and drop empty-address picks whose record has this many name-similar S1 entities


def _bucket(n: int) -> pl.Expr:
    return pl.col("s1_id").str.replace_all(r"\D", "").cast(pl.UInt64, strict=False).fill_null(0) % n


def _select(df: pl.DataFrame, rule: dict) -> pl.DataFrame:
    """Rows of (s1_id, cand_id, p) kept by rule {t, m, r}."""
    df = df.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("_rk"),
                         pl.col("p").max().over("s1_id").alias("_mx"))
    keep = pl.col("p") >= rule["t"]
    if rule.get("m"):
        keep &= pl.col("_rk") <= rule["m"]
    if rule.get("r"):
        keep &= pl.col("p") >= rule["r"] * pl.col("_mx")
    return df.filter(keep).select("s1_id", "cand_id", "p")


def write_candidates(artifacts_dir: Path, path: Path, s1: pl.DataFrame) -> Path:
    """candidate_pairs.tsv for the artifacts' own test set (the validator otherwise falls back to output/candidate_pairs.tsv,
    which belongs to the full test set)."""
    parts = [pl.read_parquet(f, columns=["s1_id", "cand_id", "rrf_rank"]).sort(["s1_id", "rrf_rank"])
               .group_by("s1_id", maintain_order=True).agg(pl.col("cand_id").unique(maintain_order=True).str.join(","))
             for f in candidate_part_files(artifacts_dir / "test")]
    c = (pl.concat(parts).group_by("s1_id", maintain_order=True).agg(pl.col("cand_id").str.join(","))   # an entity can sit in 2 parts (dense delta)
           .rename({"s1_id": "source1_entity_id", "cand_id": "candidate_entity_ids"}))
    s1.join(c, on="source1_entity_id", how="left").with_columns(pl.col("candidate_entity_ids").fill_null("")).write_csv(
        path, separator="	", quote_style="never")
    log(f"wrote {path} ({s1.height:,} rows) for the validator")
    return path


def load_rules(artifacts_dir: Path, need_llm_only: bool) -> dict:
    rule_f = ml_dir(artifacts_dir, "train") / "decision_rule.json"
    if not rule_f.exists():
        raise SystemExit(f"{rule_f} missing: run `python -m business_entity_resolution.ml decide` first")
    rules = load_json(rule_f)
    rep_f = ml_dir(artifacts_dir, "train") / "decide_report.json"
    if rep_f.exists():
        w1 = load_json(rep_f).get("by_w", {}).get("1.0")
        if w1:
            rules["llm_only"] = {"w": 1.0, "t": w1["t"], "m": w1["m"], "r": w1["r"]}
    if need_llm_only and "llm_only" not in rules:
        raise SystemExit(f"llm_only variant needs the w = 1.0 rule in {rep_f} (run `decide` with LLM scores)")
    return rules


def run(artifacts_dir: Path, output_dir: Path, test_dir: Path | None, candidate_file: Path | None, scores: Path | None,
        check_ids: bool = True, variant: str = "blend", country_variant: dict[str, str] | None = None,
        resolve_conflicts: bool = True, uniqueness: bool | None = None) -> Path:
    country_variant = country_variant or {}
    rules = load_rules(artifacts_dir, "llm_only" in (variant, *country_variant.values()))
    gbm_test = ml_dir(artifacts_dir, "test") / "gbm_test.parquet"
    uniq_f = ml_dir(artifacts_dir, "test") / "uniq_flags.parquet"
    if uniqueness is None:
        uniqueness = uniq_f.exists()
    if uniqueness and not uniq_f.exists():
        raise SystemExit(f"--uniqueness needs {uniq_f}: run `python -m business_entity_resolution.ml uniq-flags --split test`")
    uniq = pl.read_parquet(uniq_f) if uniqueness else None
    if uniq is not None:
        log(f"uniqueness rule: {uniq.height:,} empty-address candidate pairs flagged ({uniq_f.name}); add unique with p >= {UNIQ_ADD_P}, drop >= {UNIQ_DROP_N} claimants")
    if gbm_test.exists():
        picked = _pick_catboost(gbm_test, rules, scores, artifacts_dir, variant, country_variant, uniq)
    else:
        log(f"WARNING: {gbm_test} missing: using the proxy p for every test candidate (run `gbm predict --split test`)")
        picked = _pick_proxy(artifacts_dir, rules, scores)
    return _write(artifacts_dir, output_dir, picked, test_dir, candidate_file, check_ids, resolve_conflicts)


def _apply_uniqueness(picks: pl.DataFrame, dfp: pl.DataFrame, uniq_b: pl.DataFrame, stats: dict) -> pl.DataFrame:
    """Add unique empty-address records with p >= UNIQ_ADD_P; drop picks whose record has >= UNIQ_DROP_N name-similar entities."""
    if uniq_b.height == 0:
        return picks
    add = (dfp.join(uniq_b.filter(pl.col("n_name80") == 1).select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi")
              .filter(pl.col("p") >= UNIQ_ADD_P).select("s1_id", "cand_id", "p"))
    drop = uniq_b.filter(pl.col("n_name80") >= UNIQ_DROP_N).select("s1_id", "cand_id")
    n0 = picks.height
    picks = picks.join(drop, on=["s1_id", "cand_id"], how="anti")
    stats["uniq_dropped"] += n0 - picks.height
    n1 = picks.height
    picks = pl.concat([picks, add]).unique(subset=["s1_id", "cand_id"], keep="first")
    stats["uniq_added"] += picks.height - n1
    return picks


def _pick_catboost(gbm_test: Path, rules: dict, scores: Path | None, artifacts_dir: Path, variant: str, country_variant: dict[str, str],
                   uniq: pl.DataFrame | None) -> list[pl.DataFrame]:
    if rules.get("p_source") != "catboost":
        log("WARNING: decision_rule.json was not tuned on CatBoost p (re-run `decide` after `gbm predict --split valid`)")
    s = ent_var = None
    if scores is not None and scores.exists():
        s = pl.read_parquet(scores, columns=["s1_id", "cand_id", "p_llm"])
        ents = s.select("s1_id").unique().with_columns(pl.lit(variant).alias("variant"))
        if country_variant:
            ctry = pl.read_parquet(ml_dir(artifacts_dir, "test") / "gbm_entity_test.parquet", columns=["s1_id", "country"])
            ents = ents.join(ctry, on="s1_id", how="left").with_columns(
                pl.col("country").replace_strict(country_variant, default=None).fill_null(pl.col("variant")).alias("variant")).drop("country")
        ent_var = ents
        counts = dict(ents["variant"].value_counts().iter_rows())
        log(f"LLM-scored test entities: {ents.height:,} -> " + ", ".join(f"{v}: {n:,} (rule {rules[v]})" for v, n in counts.items())
            + f"; CatBoost rule {rules['gbm_only']} for the rest")
    else:
        log(f"no LLM test scores at {scores}: CatBoost-only safety submission, rule {rules['gbm_only']}")
    dense_f = gbm_test.with_name("gbm_test_dense.parquet")   # dense-recall delta (ml/dense_delta.py), new pairs only
    gbm_src = [gbm_test] + [x for x in (dense_f, gbm_test.with_name("gbm_test_dense2.parquet")) if x.exists()]
    for x in gbm_src[1:]:
        log(f"CatBoost p also from {x.name}: {pl.scan_parquet(x).select(pl.len()).collect().item():,} dense-recall pairs")
    n_rows = pl.scan_parquet(gbm_src).select(pl.len()).collect().item()
    n_b = max(1, -(-n_rows // BUCKET_ROWS))
    picked, stats = [], {"uniq_added": 0, "uniq_dropped": 0}
    for b in pbar(range(n_b), desc="export (CatBoost p)", unit="bucket"):
        df = pl.scan_parquet(gbm_src).select("s1_id", "cand_id", "p_gbm").filter(_bucket(n_b) == b).collect()
        picks, frames = [], []
        if s is None:
            dfp = df.rename({"p_gbm": "p"})
            picks.append(_select(dfp, rules["gbm_only"]))
        else:
            sb = s.filter(_bucket(n_b) == b)
            d_llm = (df.join(ent_var, on="s1_id", how="inner").join(sb, on=["s1_id", "cand_id"], how="left"))
            for v in VARIANTS:
                d_v = d_llm.filter(pl.col("variant") == v)
                if d_v.height == 0:
                    continue
                w = rules[v]["w"]
                d_v = d_v.with_columns(pl.when(pl.col("p_llm").is_not_null()).then(w * pl.col("p_llm") + (1 - w) * pl.col("p_gbm"))
                                         .otherwise(pl.col("p_gbm")).alias("p")).select("s1_id", "cand_id", "p")
                picks.append(_select(d_v, rules[v]))
                frames.append(d_v)
            d_cb = df.join(ent_var.select("s1_id"), on="s1_id", how="anti").rename({"p_gbm": "p"})
            picks.append(_select(d_cb, rules["gbm_only"]))
            frames.append(d_cb)
            dfp = pl.concat(frames)
        out = pl.concat(picks)
        if uniq is not None:
            out = _apply_uniqueness(out, dfp, uniq.filter(_bucket(n_b) == b), stats)
        picked.append(out)
    if uniq is not None:
        log(f"uniqueness rule: added {stats['uniq_added']:,} pairs, dropped {stats['uniq_dropped']:,}")
    return picked


def _pick_proxy(artifacts_dir: Path, rules: dict, scores: Path | None) -> list[pl.DataFrame]:
    """Pre-CatBoost export: blocks' p_gbm inside LLM blocks, the proxy everywhere else."""
    picked = []
    scored = pl.DataFrame(schema={"s1_id": pl.String})
    if scores is not None and scores.exists():
        s = pl.read_parquet(scores, columns=["s1_id", "cand_id", "p_llm", "p_gbm"])
        w = rules["blend"]["w"]
        s = s.with_columns((w * pl.col("p_llm") + (1 - w) * pl.col("p_gbm")).alias("p"))
        picked.append(_select(s, rules["blend"]))
        scored = s.select("s1_id").unique()
        log(f"LLM-scored test entities: {scored.height:,} (blend rule {rules['blend']})")
    else:
        log(f"no LLM test scores at {scores}: every entity uses the GBM-only rule {rules['gbm_only']}")
    for f in pbar(candidate_part_files(artifacts_dir / "test"), desc="export", unit="part"):
        lf, _ = attach_p_gbm(pl.scan_parquet(f).select(CAND_COLS), artifacts_dir, "test", source="proxy")
        df = lf.join(scored.lazy(), on="s1_id", how="anti").select("s1_id", "cand_id", pl.col("p_gbm").alias("p")).collect()
        picked.append(_select(df, rules["gbm_only"]))
    return picked


def _write(artifacts_dir: Path, output_dir: Path, picked: list[pl.DataFrame], test_dir: Path | None, candidate_file: Path | None,
           check_ids: bool, resolve_conflicts: bool = True) -> Path:
    picked = [x for x in picked if x.height] or [pl.DataFrame(schema={"s1_id": pl.String, "cand_id": pl.String, "p": pl.Float64})]
    pairs = pl.concat(picked, how="vertical_relaxed").unique(subset=["s1_id", "cand_id"], keep="first")
    if resolve_conflicts:
        n0 = pairs.height
        pairs = pairs.sort("p", descending=True, nulls_last=True).unique(subset=["cand_id"], keep="first", maintain_order=True)
        log(f"conflicts: {n0 - pairs.height:,} of {n0:,} pairs claimed a record another entity claims with a higher p -> dropped")
    matches = pairs.group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias("matched_entity_ids"))
    s1 = load_frame(artifacts_dir / "test" / "normalized_source1.pkl").select(pl.col("entity_id").alias("source1_entity_id"))
    out = (s1.join(matches.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col("matched_entity_ids").fill_null("")))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "matching_results.tsv"
    out.write_csv(path, separator="\t", quote_style="never")
    n_nonempty = int((out["matched_entity_ids"] != "").sum())
    log(f"wrote {path}: {out.height:,} S1 entities, {n_nonempty:,} with >= 1 match ({n_nonempty / max(1, out.height):.1%}), {pairs.height:,} pairs")
    if test_dir is not None:
        validator = Path(__file__).resolve().parents[3] / "utils" / "validate_submission.py"
        if candidate_file is None or not candidate_file.exists():
            candidate_file = write_candidates(artifacts_dir, output_dir / "candidate_pairs.tsv", s1)
        cmd = [sys.executable, str(validator), "--matching", str(path), "--test-dir", str(test_dir), "--candidate", str(candidate_file)]
        if check_ids:
            cmd.append("--check-ids")
        log("validator: " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        log(f"validator exit code {rc} ({'PASS' if rc == 0 else 'FAIL'})")
        if rc != 0:
            raise SystemExit(rc)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--test-dir", default="dataset/test", help="for the validator; '' to skip validation")
    p.add_argument("--candidate-file", default="output/candidate_pairs.tsv", help="must belong to the same test set; '' = write one from the artifacts' test candidates")
    p.add_argument("--scores", default=None, help="default: <artifacts>/test/ml/llm_scores_test.parquet; 'none' = stage A only "
                                                   "(the CatBoost safety submission)")
    p.add_argument("--variant", default="blend", choices=VARIANTS, help="rule for LLM-scored entities")
    p.add_argument("--country-variant", action="append", default=[], metavar="COUNTRY=VARIANT", help="per-country override, e.g. France=llm_only (repeatable)")
    p.add_argument("--no-resolve-conflicts", action="store_true", help="keep records claimed by several entities")
    p.add_argument("--uniqueness", dest="uniqueness", action="store_true", default=None, help="force the empty-address uniqueness rule")
    p.add_argument("--no-uniqueness", dest="uniqueness", action="store_false", help="skip it even when uniq_flags.parquet exists")
    p.add_argument("--no-check-ids", action="store_true", help="skip the validator's (memory-heavy) id-existence check")
    a = p.parse_args(argv)
    art = Path(a.artifacts_dir)
    scores = None if (a.scores or "").lower() == "none" else Path(a.scores) if a.scores else ml_dir(art, "test") / "llm_scores_test.parquet"
    cv = {}
    for item in a.country_variant:
        c, _, v = item.partition("=")
        if v not in VARIANTS:
            raise SystemExit(f"--country-variant {item}: variant must be one of {VARIANTS}")
        cv[c] = v
    run(art, Path(a.output_dir), Path(a.test_dir) if a.test_dir else None, Path(a.candidate_file) if a.candidate_file else None,
        scores, check_ids=not a.no_check_ids, variant=a.variant, country_variant=cv, resolve_conflicts=not a.no_resolve_conflicts,
        uniqueness=a.uniqueness)


if __name__ == "__main__":
    main()
