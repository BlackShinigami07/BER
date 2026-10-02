"""Shared CPU-side helpers for the ML stage: artifact layout and stage-A probabilities.

Stage-A probabilities come from the CatBoost model (CATBOOST_PLAN.md, ml/gbm.py: train/ml/gbm_valid.parquet,
test/ml/gbm_test.parquet with s1_id, cand_id, p_gbm) when its files exist. A provenance-only logistic proxy (ML_PLAN
"placeholder p_gbm") stands in otherwise: 2-fold cross-fitted by entity on ml_train (so p inside training blocks is
out-of-fold) and fitted on all sampled ml_train rows for ml_valid / test; coefficients in train/ml/proxy_model.json.

The LLM blocks keep the proxy p in their prompts (the stage-B model was trained with it): blocks.py asks for
source="proxy". Decisions, routing and the stage-C blend use CatBoost's p (decide.py, export_results.py). Blocks
built from CatBoost p for ml_train would need out-of-fold probabilities (gbm_oof_train.parquet, a 2-fold fit).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..io import load_frame, load_json, save_json
from ..progress import log

SUBSET_SPLIT = {"train": "train", "dev": "train", "valid": "train", "test": "test"}
GBM_FILES = {"train": ["gbm_oof_train.parquet", "gbm_valid.parquet"], "test": ["gbm_test.parquet"]}
P_SOURCES = ("auto", "proxy", "gbm")
CAND_COLS = ["s1_id", "cand_id", "src", "country", "state_match", "n_blockers", "rrf_score", "rrf_rank",
             "rank_exact", "rank_word", "rank_char_name", "rank_char_addr", "cos_exact", "cos_word", "cos_char_name", "cos_char_addr"]
PROXY_SEED = 42


def ml_dir(artifacts_dir, split: str) -> Path:
    return Path(artifacts_dir) / split / "ml"


def subset_split(subset: str) -> str:
    """Split a block subset belongs to: the four named subsets, else by prefix (test_aug0, test_job2 -> test; valid_aug -> train)."""
    return SUBSET_SPLIT.get(subset, "test" if subset.startswith("test") else "train")


def blocks_dir(artifacts_dir, subset: str) -> Path:
    return ml_dir(artifacts_dir, subset_split(subset)) / "blocks" / subset


def candidate_part_files(split_dir) -> list[Path]:
    files = sorted((Path(split_dir) / "candidate_parts").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no candidate parts under {split_dir}/candidate_parts (run the blocking stage first)")
    return files


def load_folds(artifacts_dir) -> pl.DataFrame:
    return load_frame(Path(artifacts_dir) / "train" / "splits.pkl").select(pl.col("entity_id").alias("s1_id"), "fold")


def gbm_available(artifacts_dir, split: str) -> bool:
    return all((ml_dir(artifacts_dir, split) / f).exists() for f in GBM_FILES[split])


# ---------------------------------------------------------------- provenance proxy for p_gbm
def _proxy_features() -> list[pl.Expr]:
    nb = pl.col("n_blockers").cast(pl.Int32)
    f = [(nb == 1), (nb == 2), (nb == 3), (nb >= 4),
         pl.col("rrf_score").cast(pl.Float64) * 10, (pl.col("rrf_rank").cast(pl.Float64) + 1).log(),
         pl.col("state_match") == "same", pl.col("state_match") == "diff", pl.col("src") == "S3"]
    for b in ("exact", "word", "char_name", "char_addr"):
        f += [pl.col(f"rank_{b}").is_not_null(), pl.col(f"cos_{b}").cast(pl.Float64).fill_null(0.0)]
    return [e.cast(pl.Float64).alias(f"x{i}") for i, e in enumerate(f)]


def _proxy_logit(coef: dict) -> pl.Expr:
    feats = _proxy_features()
    z = pl.lit(coef["intercept"])
    for w, e in zip(coef["coef"], feats):
        z = z + w * e
    return z


def _half() -> pl.Expr:
    """Deterministic 2-fold split of S1 entities by the parity of the numeric id (ids are random; stable across runs
    and library versions, unlike polars' hash)."""
    return (pl.col("s1_id").str.extract(r"(\d+)$", 1).cast(pl.Int64, strict=False).fill_null(PROXY_SEED) % 2).cast(pl.Int8)


def fit_proxy(artifacts_dir, max_entities: int = 60_000, seed: int = 42) -> dict:
    """Fit the proxy on a sample of ml_train entities: models 'h0' (fit on half 0), 'h1' (half 1) and 'full'."""
    from sklearn.linear_model import LogisticRegression
    out = ml_dir(artifacts_dir, "train") / "proxy_model.json"
    folds = load_folds(artifacts_dir)
    ids = folds.filter(pl.col("fold") == "ml_train")["s1_id"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = pl.DataFrame({"s1_id": rng.choice(ids, size=min(max_entities, len(ids)), replace=False)})
    lf = pl.scan_parquet([str(f) for f in candidate_part_files(Path(artifacts_dir) / "train")])
    df = (lf.join(pick.lazy(), on="s1_id", how="semi")
            .select(_half().alias("half"), pl.col("label").cast(pl.Int8), *_proxy_features()).collect())
    X = df.select([c for c in df.columns if c.startswith("x")]).to_numpy()
    y = df["label"].to_numpy()
    h = df["half"].to_numpy()
    models = {}
    for name, m in (("h0", h == 0), ("h1", h == 1), ("full", np.ones(len(y), dtype=bool))):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(X[m], y[m])
        models[name] = {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    res = {"models": models, "n_rows": int(len(y)), "positive_rate": float(y.mean()), "seed": seed,
           "note": "provenance-only placeholder for stage-A p_gbm until WP-A lands (ml/common.py)"}
    save_json(res, out)
    log(f"proxy p_gbm fitted on {len(y):,} ml_train rows (positive rate {y.mean():.3f}) -> {out}")
    return res


def load_proxy(artifacts_dir) -> dict:
    f = ml_dir(artifacts_dir, "train") / "proxy_model.json"
    return load_json(f) if f.exists() else fit_proxy(artifacts_dir)


def attach_p_gbm(lf: pl.LazyFrame, artifacts_dir, split: str, folds: pl.DataFrame | None = None,
                 source: str = "auto") -> tuple[pl.LazyFrame, str]:
    """Add `p_gbm` (and `fold` for the train split) to a candidate LazyFrame. Returns (frame, source name).
    source: "auto" = CatBoost when its files exist else the proxy; "proxy"; "gbm" (error when missing)."""
    if source not in P_SOURCES:
        raise ValueError(f"source must be one of {P_SOURCES}")
    if split == "train":
        folds = load_folds(artifacts_dir) if folds is None else folds
        lf = lf.join(folds.lazy(), on="s1_id", how="left")
    if source == "gbm" and not gbm_available(artifacts_dir, split):
        raise FileNotFoundError(f"CatBoost probabilities {GBM_FILES[split]} missing under {ml_dir(artifacts_dir, split)}")
    if source != "proxy" and gbm_available(artifacts_dir, split):
        gbm = pl.concat([pl.scan_parquet(ml_dir(artifacts_dir, split) / f).select("s1_id", "cand_id", pl.col("p_gbm").cast(pl.Float32))
                         for f in GBM_FILES[split]])
        lf = lf.join(gbm, on=["s1_id", "cand_id"], how="left")
        return lf.with_columns(pl.col("p_gbm").fill_null(0.0)), "gbm"
    models = load_proxy(artifacts_dir)["models"]
    sig = lambda z: 1.0 / (1.0 + (-z).exp())
    if split == "train":
        # out-of-fold for ml_train: an entity in half 0 is scored by the model fitted on half 1, and vice versa
        p = (pl.when(pl.col("fold") != "ml_train").then(sig(_proxy_logit(models["full"])))
               .when(_half() == 0).then(sig(_proxy_logit(models["h1"])))
               .otherwise(sig(_proxy_logit(models["h0"]))))
    else:
        p = sig(_proxy_logit(models["full"]))
    return lf.with_columns(p.cast(pl.Float32).alias("p_gbm")), "proxy"
