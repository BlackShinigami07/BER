"""WP-A pair features for the stage-A CatBoost model (CATBOOST_PLAN.md section 3, ARTIFACTS.md section 6).

Per (S1 entity, candidate) pair: the blocking provenance (the RRF features: n_blockers, rrf_score, rrf_rank and every
blocker's rank + cosine, NA encoded explicitly), string similarities on the normalised name and address (rapidfuzz,
vectorised with process.cpdist), numeric / house-number agreement, scripts, empties, lengths, per-entity context
(how the pair compares with the entity's other candidates), then the orthogonal families of ml/features_extra.py.
All float32; missing = null (NaN for CatBoost) where a comparison is undefined (e.g. an empty address: rapidfuzz scores
"" vs "" as 100, which would fake a perfect match).

Contract: the frame given to `pair_features` is entity-complete (every candidate of each of its S1 entities), and the
part-level family columns (candidate graph) were computed on the whole candidate part; ml/gbm_features.py guarantees
both. Labels are never read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..io import load_frame
from .features_extra import (PRODUCTION_FAMILIES, ExtraLookups, add_families, add_part_families, cpdist, family_columns,
                             part_columns)

BLOCKERS = ("exact", "word", "char_name", "char_addr")
CAND_COLS = ["s1_id", "cand_id", "src", "country", "state_match", "n_blockers", "rrf_score", "rrf_rank"] + \
            [f"rank_{b}" for b in BLOCKERS] + [f"cos_{b}" for b in BLOCKERS]
TEXT_FIELDS = ("full", "core", "norm", "comps", "raw", "script")     # per side: q_* (S1) and c_* (candidate)
KEY_COLS = ("s1_id", "cand_id", "country", "label", "fold", "weight")  # carried through, never used as features

BASE_FEATURES = (
    # blocking provenance (RRF fusion + per-blocker evidence)
    ["n_blockers", "rrf_score", "rrf_rank"] + [f"rank_{b}" for b in BLOCKERS] + [f"cos_{b}" for b in BLOCKERS] +
    [f"has_{b}" for b in BLOCKERS] + ["state_same", "state_diff", "src_s3", "is_india"] +
    # names
    ["core_ratio", "core_tset", "core_partial", "core_jw", "full_ratio", "full_tsort", "core_equal", "first_tok_equal",
     "q_core_len", "c_core_len", "core_len_diff", "c_indic", "c_other_script", "q_nonlatin"] +
    # addresses
    ["addr_tset", "addr_ratio", "addr_tok_jacc", "num_jacc", "first_num_equal", "q_addr_empty", "c_addr_empty"] +
    # per-entity context
    ["ctx_n_cands", "ctx_nb3", "ctx_rrf_rel", "ctx_core_tset_gap", "ctx_core_tset_rank", "ctx_addr_tset_gap", "ctx_cos_name_rank"]
)
EXTRA_FEATURES = tuple(family_columns(PRODUCTION_FAMILIES))
FEATURES = BASE_FEATURES + list(EXTRA_FEATURES)          # the production model schema, in this order


def features_for(families) -> list[str]:
    """Model columns for a family selection (BASE_FEATURES first, then the families' columns)."""
    return BASE_FEATURES + family_columns(families)


# ---------------------------------------------------------------- text lookups
def text_side(split_dir, side: str) -> pl.DataFrame:
    """Normalised text of one side keyed by s1_id ("q") or cand_id ("c"), from normalized_*.pkl + final_addr_*.pkl."""
    split_dir = Path(split_dir)
    norm_files, addr_file, key = ((["normalized_source1.pkl"], "final_addr_source1.pkl", "s1_id") if side == "q" else
                                  (["normalized_source2.pkl", "normalized_source3.pkl"], "final_addr_source23.pkl", "cand_id"))
    norm = pl.concat([load_frame(split_dir / f).select("entity_id", "name_full", "name_core", "name_script", "addr_comps",
                                                       "business_address") for f in norm_files])
    addr = load_frame(split_dir / addr_file).select("entity_id", "addr_norm")
    return (norm.join(addr, on="entity_id", how="left")
                .select(pl.col("entity_id").alias(key),
                        pl.col("name_full").fill_null("").alias(f"{side}_full"),
                        pl.col("name_core").fill_null("").alias(f"{side}_core"),
                        pl.col("addr_norm").fill_null("").alias(f"{side}_norm"),
                        pl.col("addr_comps").fill_null("").alias(f"{side}_comps"),
                        pl.col("business_address").fill_null("").alias(f"{side}_raw"),
                        pl.col("name_script").fill_null("latin").alias(f"{side}_script")))


def load_lookups(split_dir) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(s1, pool) text tables fully in memory (mini artifacts, tests; the full pipeline streams them, gbm_features)."""
    return text_side(split_dir, "q"), text_side(split_dir, "c")


class FrameText:
    """In-memory text accessor with the same interface as gbm_features.TextStore."""

    def __init__(self, s1: pl.DataFrame, pool: pl.DataFrame):
        self.s1_df, self.pool_df = s1, pool

    def s1(self, ids: pl.Series, cols=None) -> pl.DataFrame:
        d = self.s1_df.join(pl.DataFrame({"s1_id": ids.unique()}), on="s1_id", how="semi")
        return d if cols is None else d.select("s1_id", *cols)

    def pool(self, ids: pl.Series, cols=None) -> pl.DataFrame:
        d = self.pool_df.join(pl.DataFrame({"cand_id": ids.unique()}), on="cand_id", how="semi")
        return d if cols is None else d.select("cand_id", *cols)

    pool_text = pool


def join_text(c: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    df = c.join(s1, on="s1_id", how="left").join(pool, on="cand_id", how="left")
    return df.with_columns([pl.col(f"{p}_{f}").fill_null("") for p in ("q", "c") for f in TEXT_FIELDS if f != "script"] +
                           [pl.col(f"{p}_script").fill_null("latin") for p in ("q", "c")])


# ---------------------------------------------------------------- the 47 base features
def base_features(df: pl.DataFrame) -> pl.DataFrame:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    col = lambda name: df[name].to_numpy()
    qc, cc, qf, cf, qa, ca = (col(x) for x in ("q_core", "c_core", "q_full", "c_full", "q_norm", "c_norm"))
    both = ~((qa == "") | (ca == ""))
    nan = np.float32(np.nan)
    text = {
        "core_ratio": cpdist(fuzz.ratio, qc, cc), "core_tset": cpdist(fuzz.token_set_ratio, qc, cc),
        "core_partial": cpdist(fuzz.partial_ratio, qc, cc), "core_jw": cpdist(JaroWinkler.normalized_similarity, qc, cc, scale=1.0),
        "full_ratio": cpdist(fuzz.ratio, qf, cf), "full_tsort": cpdist(fuzz.token_sort_ratio, qf, cf),
        "addr_tset": np.where(both, cpdist(fuzz.token_set_ratio, qa, ca), nan),
        "addr_ratio": np.where(both, cpdist(fuzz.ratio, qa, ca), nan),
    }
    df = df.with_columns([pl.Series(k, v.astype(np.float32)).fill_nan(None) for k, v in text.items()])   # NaN -> null
    tok = lambda x: pl.col(x).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    num = lambda x: pl.col(x).str.extract_all(r"\d+")
    jacc = lambda a, b: (a.list.set_intersection(b).list.len() / a.list.set_union(b).list.len()).cast(pl.Float32)
    nb = pl.col("n_blockers").cast(pl.Float32)
    df = df.with_columns(
        nb.alias("n_blockers"), pl.col("rrf_score").cast(pl.Float32), pl.col("rrf_rank").cast(pl.Float32),
        *[pl.col(f"rank_{b}").cast(pl.Float32).fill_null(999.0).alias(f"rank_{b}") for b in BLOCKERS],
        *[pl.col(f"cos_{b}").cast(pl.Float32).fill_null(0.0).alias(f"cos_{b}") for b in BLOCKERS],
        *[pl.col(f"rank_{b}").is_not_null().cast(pl.Float32).alias(f"has_{b}") for b in BLOCKERS],
        (pl.col("state_match") == "same").cast(pl.Float32).alias("state_same"),
        (pl.col("state_match") == "diff").cast(pl.Float32).alias("state_diff"),
        (pl.col("src") == "S3").cast(pl.Float32).alias("src_s3"),
        (pl.col("country") == "India").cast(pl.Float32).alias("is_india"),
        (pl.col("q_core") == pl.col("c_core")).cast(pl.Float32).alias("core_equal"),
        (pl.col("q_core").str.split(" ").list.first() == pl.col("c_core").str.split(" ").list.first()).cast(pl.Float32).alias("first_tok_equal"),
        pl.col("q_core").str.len_chars().cast(pl.Float32).alias("q_core_len"),
        pl.col("c_core").str.len_chars().cast(pl.Float32).alias("c_core_len"),
        (pl.col("q_core").str.len_chars().cast(pl.Int32) - pl.col("c_core").str.len_chars().cast(pl.Int32)).abs().cast(pl.Float32).alias("core_len_diff"),
        (pl.col("c_script") == "indic").cast(pl.Float32).alias("c_indic"),
        (pl.col("c_script") == "other").cast(pl.Float32).alias("c_other_script"),
        (pl.col("q_script") != "latin").cast(pl.Float32).alias("q_nonlatin"),
        pl.when((pl.col("q_norm") != "") & (pl.col("c_norm") != "")).then(jacc(tok("q_norm"), tok("c_norm"))).otherwise(None).alias("addr_tok_jacc"),
        pl.when((num("q_norm").list.len() > 0) & (num("c_norm").list.len() > 0)).then(jacc(num("q_norm"), num("c_norm"))).otherwise(None).alias("num_jacc"),
        pl.when((num("q_norm").list.len() > 0) & (num("c_norm").list.len() > 0))
          .then((num("q_norm").list.first() == num("c_norm").list.first()).cast(pl.Float32)).otherwise(None).alias("first_num_equal"),
        (pl.col("q_norm") == "").cast(pl.Float32).alias("q_addr_empty"),
        (pl.col("c_norm") == "").cast(pl.Float32).alias("c_addr_empty"),
    )
    g = "s1_id"
    return df.with_columns(
        pl.len().over(g).cast(pl.Float32).alias("ctx_n_cands"),
        (pl.col("n_blockers") >= 3).sum().over(g).cast(pl.Float32).alias("ctx_nb3"),
        (pl.col("rrf_score") / pl.col("rrf_score").max().over(g)).cast(pl.Float32).alias("ctx_rrf_rel"),
        (pl.col("core_tset") - pl.col("core_tset").max().over(g)).cast(pl.Float32).alias("ctx_core_tset_gap"),
        pl.col("core_tset").rank("min", descending=True).over(g).cast(pl.Float32).alias("ctx_core_tset_rank"),
        (pl.col("addr_tset") - pl.col("addr_tset").max().over(g)).cast(pl.Float32).alias("ctx_addr_tset_gap"),
        pl.col("cos_char_name").rank("min", descending=True).over(g).cast(pl.Float32).alias("ctx_cos_name_rank"),
    )


def pair_features(c: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, *, extra_lookups: ExtraLookups | None = None,
                  families=PRODUCTION_FAMILIES, columns: list[str] | None = None, timings: dict | None = None) -> pl.DataFrame:
    """Features for the entity-complete candidate rows `c`; returns c's KEY_COLS that are present and
    `columns` (default: BASE_FEATURES + the families' columns). Part-level family columns are taken from `c` when the
    caller computed them on the whole part (gbm_features); otherwise `c` itself is treated as the part."""
    import time
    keep = [k for k in KEY_COLS if k in c.columns]
    columns = features_for(families) if columns is None else columns
    if any(x not in c.columns for x in part_columns(families)):
        c = add_part_families(c, FrameText(s1, pool), families, timings)
    t = time.perf_counter()
    df = base_features(join_text(c, s1, pool))
    if timings is not None:
        timings["base"] = timings.get("base", 0.0) + time.perf_counter() - t
    df = add_families(df, extra_lookups, families, timings)
    return df.select(keep + [pl.col(f).cast(pl.Float32) for f in columns])
