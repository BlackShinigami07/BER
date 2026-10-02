"""WP-0 harness (CATBOOST_PLAN.md section 6): cached feature frame + the measured evaluation protocol on mini.

The cache (`<artifacts>/train/ml/feat_cache.parquet`) holds every registered feature family for every train row,
computed by the production code path (gbm_features.part_frames: part-level graph on whole parts, entity-complete
chunks), so an ablation costs only CatBoost time. It is rebuilt when the feature code changes (source hash).

Protocol (identical to the section-0 measurements): fit = ml_train entities whose numeric id % 10 != 0, early
stopping = ml_train entities with id % 10 == 0, evaluation = all ml_valid entities, metric = per-entity macro F0.5
with the singleton rule and n_truth including blocking misses, best decision rule from decide.search; paired
bootstrap over entities for differences between feature sets.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..io import load_frame, load_json, save_json
from ..progress import fmt_secs, log, pbar
from .common import candidate_part_files, load_folds, ml_dir
from .features_extra import FAMILIES

CACHE = "feat_cache.parquet"
_SRC = [Path(__file__).with_name(n) for n in ("features.py", "features_extra.py", "gbm_features.py")]


def code_hash() -> str:
    h = hashlib.sha1()
    for f in _SRC:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load_cache(artifacts_dir, *, rebuild: bool = False, chunk_rows: int = 1_000_000) -> tuple[pl.DataFrame, dict]:
    """(frame with keys + label + fold + every family's features, meta with per-family feature seconds)."""
    from .gbm_features import open_split, part_frames
    d = ml_dir(artifacts_dir, "train")
    f, meta_f = d / CACHE, d / "feat_cache.json"
    families = tuple(FAMILIES)
    if f.exists() and meta_f.exists() and not rebuild:
        meta = load_json(meta_f)
        if meta.get("code") == code_hash() and meta.get("families") == list(families):
            return pl.read_parquet(f), meta
        log("feature code changed since the cache was built: rebuilding")
    t0 = time.time()
    text, lookups = open_split(artifacts_dir, "train")
    timings: dict[str, float] = {}
    frames = [fr for part in pbar(candidate_part_files(Path(artifacts_dir) / "train"), desc="feature cache", unit="part")
              for fr in part_frames(part, text, lookups, families=families, label=True, chunk_rows=chunk_rows, timings=timings)]
    df = (pl.concat(frames).join(load_folds(artifacts_dir), on="s1_id", how="left")
            .with_columns(pl.col("label").cast(pl.Int8)).sort(["s1_id", "cand_id"]))
    df.write_parquet(f, compression="zstd")
    meta = {"code": code_hash(), "families": list(families), "rows": df.height, "seconds": round(time.time() - t0, 1),
            "feature_seconds": {k: round(v, 3) for k, v in timings.items()}}
    save_json(meta, meta_f)
    log(f"feature cache: {df.height:,} rows x {df.width} columns in {fmt_secs(time.time() - t0)} -> {f}")
    return df, meta


@dataclass
class Protocol:
    tr: pl.DataFrame
    es: pl.DataFrame
    va: pl.DataFrame
    off: np.ndarray          # block offsets of va (sorted by s1_id)
    n_truth: np.ndarray      # per va entity, including blocking misses
    y_tr: np.ndarray
    y_es: np.ndarray
    y_va: np.ndarray


def protocol(df: pl.DataFrame, artifacts_dir) -> Protocol:
    h = pl.col("s1_id").str.extract(r"(\d+)$", 1).cast(pl.Int64) % 10
    df = df.sort(["s1_id", "cand_id"])
    tr = df.filter((pl.col("fold") == "ml_train") & (h != 0))
    es = df.filter((pl.col("fold") == "ml_train") & (h == 0))
    va = df.filter(pl.col("fold") == "ml_valid")
    off, n_truth = valid_blocks(va, artifacts_dir)
    return Protocol(tr, es, va, off, n_truth, tr["label"].to_numpy(), es["label"].to_numpy(), va["label"].to_numpy())


def valid_blocks(va: pl.DataFrame, artifacts_dir) -> tuple[np.ndarray, np.ndarray]:
    """Offsets of the entity blocks of a frame sorted by s1_id and each entity's n_truth (positives + blocking misses)."""
    missed_f = Path(artifacts_dir) / "train" / "missed_pairs.pkl"
    missed = (load_frame(missed_f).group_by("s1_id").agg(pl.len().alias("n_missed")) if missed_f.exists()
              else pl.DataFrame(schema={"s1_id": pl.String, "n_missed": pl.UInt32}))
    cnt = (va.group_by("s1_id", maintain_order=True).agg(pl.len().alias("n"), pl.col("label").cast(pl.Int64).sum().alias("pos"))
             .join(missed, on="s1_id", how="left", maintain_order="left"))
    off = np.zeros(cnt.height + 1, dtype=np.int64)
    np.cumsum(cnt["n"].to_numpy(), out=off[1:])
    return off, (cnt["pos"].to_numpy() + cnt["n_missed"].fill_null(0).to_numpy()).astype(np.int64)


def catboost_params(seed: int, *, iterations: int = 4000, lr: float = 0.05, depth: int = 8, od_wait: int = 200,
                    border_count: int | None = None, task_type: str = "GPU", l2_leaf_reg: float | None = None) -> dict:
    p = dict(iterations=iterations, learning_rate=lr, depth=depth, loss_function="Logloss", eval_metric="Logloss",
             od_type="Iter", od_wait=od_wait, random_seed=seed, verbose=0, task_type=task_type, thread_count=-1,
             allow_writing_files=False)
    if task_type == "GPU":
        p["devices"] = "0"
    if border_count:
        p["border_count"] = border_count
    if l2_leaf_reg is not None:
        p["l2_leaf_reg"] = l2_leaf_reg
    return p


def run_catboost(P: Protocol, cols: list[str], seeds=(0, 1, 2), **kw) -> dict:
    """Per-entity F0.5 (each seed with its own best rule), averaged over seeds; also the seed-averaged probability."""
    import catboost as cb
    from .decide import search
    from .llm.metrics import apply_rule, block_f05
    X = lambda d: d.select(cols).to_numpy().astype(np.float32)
    Xtr, Xes, Xva = X(P.tr), X(P.es), X(P.va)
    per, probs, trees, secs = [], [], [], []
    for sd in seeds:
        t = time.time()
        m = cb.CatBoostClassifier(**catboost_params(sd, **kw)).fit(Xtr, P.y_tr, eval_set=(Xes, P.y_es), use_best_model=True)
        p = m.predict_proba(Xva)[:, 1]
        rule, _ = search(p, P.y_va, P.off, P.n_truth)
        per.append(block_f05(apply_rule(p, P.off, rule["t"], rule["m"] or None, rule["r"]), P.y_va, P.off, P.n_truth))
        probs.append(p)
        trees.append(m.get_best_iteration() + 1)
        secs.append(time.time() - t)
    per_entity = np.mean(per, axis=0)
    return {"per_entity": per_entity, "f05": float(per_entity.mean()), "f05_by_seed": [float(x.mean()) for x in per],
            "p": np.mean(probs, axis=0), "trees": trees, "fit_seconds": round(float(np.mean(secs)), 1), "n_features": len(cols)}


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of (a - b) over entities with a percentile-bootstrap 95% CI."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}
