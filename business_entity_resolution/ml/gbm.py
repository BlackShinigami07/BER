"""WP-3 / WP-4: the stage-A CatBoost classifier (CATBOOST_PLAN.md sections 4, 5, 6).

    python -m business_entity_resolution.ml gbm fit     --artifacts-dir artifacts        # 3 seeds, GPU, early stopping
    python -m business_entity_resolution.ml gbm predict --artifacts-dir artifacts --split valid
    python -m business_entity_resolution.ml gbm predict --artifacts-dir artifacts --split test
    python -m business_entity_resolution.ml gbm report  --artifacts-dir artifacts

fit: reads train/ml/gbm_fit.parquet (gbm_features build). Two loaders, same model: `numpy` (F-ordered float32 matrix,
zero-copy into a raw CatBoost Pool that GPU training quantises itself: peak ~1.25x the float32 matrix) and `file`
(WP-3b: the fit rows are written to a TSV and quantised by catboost.utils.quantize without ever holding the float32
matrix: peak ~quantised pool + parse buffers; the quantised pool is saved to train/ml/gbm_fit.quantized and reused
by later fits of the same fit file). `auto` picks numpy when it fits the available RAM with a margin, else file.
Never Pool.quantize() on an in-memory pool before GPU training: with CatBoost 1.2.10 that silently yields a broken
model once the pool is large (measured: 448k rows, early-stop AUC 0.76 vs 0.9999; CPU training and the file path
are fine), so every fit also passes a sanity gate on the early-stop holdout before its models are saved. Early stopping on the fit
entities' 5% holdout (od_wait 200, Logloss); CatBoost GPU depth 8, lr 0.05, <= 6000 trees, l2 3, 254 borders;
seeds 42/43/44, saved as train/ml/gbm_model_{seed}.cbm + gbm_model.json (schema, params, trees, importances, log).
Calibration: none unless the ensemble's weighted ECE on the early-stopping holdout is > 0.01 (then isotonic, fitted
on that holdout, never on ml_valid).

predict: streams the split's candidate parts through the same feature code (entity-complete chunks), averages the
seeds' probabilities, writes per-part files (resumable) and then
    valid: train/ml/gbm_valid.parquet  (s1_id, cand_id, p_gbm, label, country, c_indic, p_s<seed>...)
           train/ml/gbm_rule.json      (the decision rule tuned on ALL of ml_valid by decide.search)
    test : test/ml/gbm_test.parquet    (s1_id, cand_id, p_gbm)
    both : <split>/ml/gbm_entity_<split>.parquet: one row per S1 entity with u = sum_c min(p, 1-p), max_p,
           n_band = #{0.05 < p < 0.95}, n_over_t, n_pred (decision-set size under the tuned rule), sorted by
           route_rank (0 = first to route) with u_rank (0 = most uncertain); test also carries the France check.
report (WP-4): gbm_metrics.json + printed tables (section 6 WP-4).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..io import load_frame, load_json, save_json
from ..progress import available_ram_gb, fmt_secs, log, pbar, rss_gb, stage
from .common import candidate_part_files, ml_dir
from .gbm_features import id_number, iter_feature_frames, mix64, pair_hash

SEEDS = (42, 43, 44)
PARAMS = dict(task_type="GPU", depth=8, learning_rate=0.05, iterations=6000, l2_leaf_reg=3.0, border_count=254,
              loss_function="Logloss", eval_metric="Logloss", od_type="Iter", od_wait=200)
ECE_MAX = 0.01
FEATSAMPLE_MOD = {"valid": 64, "test": 512}      # rows kept for the train/test drift check (hash-sampled)
U_TABLE = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


# ---------------------------------------------------------------- small shared pieces
def model_manifest(artifacts_dir) -> dict:
    f = ml_dir(artifacts_dir, "train") / "gbm_model.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: run `python -m business_entity_resolution.ml gbm fit` first")
    return load_json(f)


def weighted_ece(p, y, w=None, bins: int = 15) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    sw = np.bincount(idx, weights=w, minlength=bins)
    sp = np.bincount(idx, weights=w * p, minlength=bins)
    sy = np.bincount(idx, weights=w * y, minlength=bins)
    ok = sw > 0
    return float(np.abs(sp[ok] - sy[ok]).sum() / max(sw.sum(), 1e-12))


def apply_calibration(p: np.ndarray, cal: dict | None) -> np.ndarray:
    if not cal:
        return p
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"])).astype(p.dtype)


class GpuSampler:
    """Samples nvidia-smi every few seconds while training: proof that the GPU is used, and its peak memory."""

    def __init__(self, every_s: float = 5.0):
        self.every, self.samples, self._stop = every_s, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                if out:
                    name, util, mem = [x.strip() for x in out[0].split(",")]
                    self.samples.append((name, float(util), float(mem)))
            except Exception:
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=15)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        return {"available": True, "gpu": self.samples[0][0], "max_util_pct": max(s[1] for s in self.samples),
                "mean_util_pct": float(np.mean([s[1] for s in self.samples])), "max_mem_mb": max(s[2] for s in self.samples),
                "n_samples": len(self.samples)}


# ---------------------------------------------------------------- fit (WP-3)
def _fit_meta(artifacts_dir) -> dict:
    f = ml_dir(artifacts_dir, "train") / "gbm_fit_config.json"
    meta = load_json(f) if f.exists() else {}
    if not meta.get("done"):
        raise SystemExit(f"{f} missing or incomplete: run `python -m business_entity_resolution.ml gbm-features build` first")
    return meta


def _subsample_expr(keep: float, seed: int) -> pl.Expr | None:
    """Entity-level (by hash) subsample of the fit rows when the matrix exceeds the RAM budget."""
    return None if keep >= 1.0 else pl.col("_h") < int(keep * 1_000_000)


def _with_hash(lf: pl.LazyFrame, seed: int) -> pl.LazyFrame:
    return lf.with_columns(pl.col("s1_id").map_batches(lambda s: pl.Series(mix64(id_number(s), seed + 7) % np.uint64(1_000_000)),
                                                        return_dtype=pl.UInt64).alias("_h"))


def _fit_rows(fit_path: Path, fold: str, keep: float, seed: int) -> pl.LazyFrame:
    """Rows of one fold of the fit file. The fold filter goes first (pushed into the parquet scan); the entity hash
    (a Python UDF, an optimisation barrier) is only added when a RAM subsample (keep < 1) is needed."""
    lf = pl.scan_parquet(fit_path).filter(pl.col("fold") == fold)
    if keep < 1.0:
        lf = _with_hash(lf, seed).filter(_subsample_expr(keep, seed))
    return lf


def _load_numpy(fit_path: Path, feats: list[str], fold: str, n_rows: int, keep: float, seed: int):
    """F-ordered float32 matrix (zero-copy into CatBoost) + label + weight for one fold, filled batch by batch."""
    X = np.empty((n_rows, len(feats)), dtype=np.float32, order="F")
    y = np.empty(n_rows, dtype=np.int8)
    w = np.empty(n_rows, dtype=np.float32)
    lf = _fit_rows(fit_path, fold, keep, seed)
    i = 0
    for batch in lf.select("label", "weight", *feats).collect_batches(chunk_size=500_000):
        n = batch.height
        X[i:i + n] = batch.select(feats).to_numpy()
        y[i:i + n] = batch["label"].to_numpy()
        w[i:i + n] = batch["weight"].to_numpy()
        i += n
    assert i == n_rows, (i, n_rows)
    return X, y, w


def _fit_pool(fit_path: Path, feats: list[str], loader: str, n_fit: int, keep: float, seed: int, border_count: int,
                    out_dir: Path, fingerprint: dict):
    """The fit pool: raw (numpy loader) or file-quantised (built or reused). Returns (pool, loader used, seconds)."""
    from catboost import Pool, utils
    qpath = out_dir / "gbm_fit.quantized"
    qmeta = out_dir / "gbm_fit.quantized.json"
    fp = {"fit": fingerprint, "keep": keep, "border_count": border_count, "features": feats, "n_fit": n_fit, "quantizer": "utils.quantize(file)"}
    if qpath.exists() and qmeta.exists() and load_json(qmeta) == fp:
        t = time.time()
        pool = Pool(f"quantized://{qpath}")
        log(f"reusing the quantised fit pool {qpath} ({pool.num_row():,} rows) in {time.time() - t:.0f}s")
        return pool, "cached", time.time() - t
    t0 = time.time()
    if loader == "numpy":   # raw zero-copy pool: GPU training quantises it (see the module docstring for why not quantize())
        X, y, w = _load_numpy(fit_path, feats, "fit", n_fit, keep, seed)
        log(f"fit matrix in RAM: {X.nbytes / 2**30:.2f} GB float32 (raw pool, quantised by the GPU trainer); peak RSS {rss_gb():.2f} GB")
        return Pool(X, y, weight=w, feature_names=feats), "numpy", time.time() - t0
    else:
        tsv, cd = out_dir / "gbm_fit.tsv", out_dir / "gbm_fit.cd"
        cd.write_text("0\tLabel\n1\tWeight\n" + "".join(f"{i + 2}\tNum\t{f}\n" for i, f in enumerate(feats)), encoding="utf-8")
        lf = _fit_rows(fit_path, "fit", keep, seed)
        with stage(f"write {n_fit:,} fit rows to {tsv.name}"):
            lf.select(pl.col("label").cast(pl.Int8), "weight", *feats).sink_csv(tsv, separator="\t", include_header=False, null_value="nan")
        log(f"{tsv.name}: {tsv.stat().st_size / 2**30:.1f} GB")
        with stage("catboost.utils.quantize (file-based, WP-3b)"):
            pool = utils.quantize(str(tsv), column_description=str(cd), border_count=border_count, thread_count=-1)
        tsv.unlink(missing_ok=True)
    log(f"quantised fit pool: {pool.num_row():,} rows x {pool.num_col()} features ({loader}) in {fmt_secs(time.time() - t0)}; "
        f"peak RSS {rss_gb():.2f} GB")
    try:
        pool.save(str(qpath))
        save_json(fp, qmeta)
    except Exception as e:   # reuse is an optimisation only
        log(f"could not save the quantised pool ({type(e).__name__}: {e})")
    return pool, loader, time.time() - t0


def _sanity_gate(p_es: list, y: np.ndarray, w: np.ndarray, seeds: list) -> None:
    """Refuse models that did not learn: every seed must beat the constant predictor's weighted logloss by 5x and reach
    AUC 0.98 on the early-stop holdout (the stage-A models are at ~0.004 logloss / 0.9999 AUC)."""
    from .llm.metrics import roc_auc
    rate = float(np.sum(w * y) / np.sum(w))
    const = -float(np.sum(w * (y * np.log(rate) + (1 - y) * np.log(1 - rate))) / np.sum(w))
    for s, p in zip(seeds, p_es):
        q = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
        ll = -float(np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))) / np.sum(w))
        auc = roc_auc(p, y)
        if not (ll < const / 5 and auc > 0.98):
            raise RuntimeError(f"seed {s}: early-stop weighted logloss {ll:.4f} (constant predictor {const:.4f}), AUC {auc:.4f}: "
                               "the model did not learn; refusing to save it (see the gbm.py docstring on quantised pools)")
    log(f"sanity gate passed: every seed beats the constant predictor ({const:.4f} weighted logloss) by > 5x with AUC > 0.98")


def fit(artifacts_dir, *, seeds=SEEDS, loader: str = "auto", max_fit_gb: float = 6.5, iterations: int = PARAMS["iterations"],
        task_type: str = "GPU", seed: int = 42, refit: bool = False) -> dict:
    import catboost as cb
    t0 = time.time()
    art = Path(artifacts_dir)
    out = ml_dir(art, "train")
    meta = _fit_meta(art)
    fit_path = out / "gbm_fit.parquet"
    feats = list(meta["fingerprint"]["features"])
    manifest_f = out / "gbm_model.json"
    if manifest_f.exists() and not refit:
        old = load_json(manifest_f)
        if (old.get("fit_fingerprint") == meta["fingerprint"] and old.get("seeds") == list(seeds)
                and all((out / f).exists() for f in old.get("models", []))):
            log(f"{manifest_f} is up to date (same fit matrix, seeds {list(seeds)}, {len(old['models'])} model files): "
                "nothing to train; pass --refit to train again")
            return old
    counts = dict(pl.scan_parquet(fit_path).group_by("fold").len().collect().iter_rows())
    n_fit_all, n_es = int(counts.get("fit", 0)), int(counts.get("early_stop", 0))
    gb = n_fit_all * len(feats) * 4 / 2**30
    keep = min(1.0, max_fit_gb / gb) if gb > max_fit_gb else 1.0
    if keep < 1.0:
        log(f"WARNING: fit matrix {gb:.2f} GB > budget {max_fit_gb} GB: keeping {keep:.1%} of the fit entities (by hash)")
    n_fit = n_fit_all if keep >= 1.0 else int(_fit_rows(fit_path, "fit", keep, seed).select(pl.len()).collect().item())
    need = n_fit * len(feats) * 4 * 1.3 / 2**30
    avail = available_ram_gb()
    if loader == "auto":
        loader = "numpy" if avail is not None and need < avail - 1.5 else "file"
    log(f"fit rows {n_fit:,} (+{n_es:,} early-stop) x {len(feats)} features = {n_fit * len(feats) * 4 / 2**30:.2f} GB float32; "
        f"numpy path needs ~{need:.1f} GB, available {avail if avail is None else round(avail, 1)} GB -> loader {loader}")
    border = PARAMS["border_count"]
    pool, loader_used, load_s = _fit_pool(fit_path, feats, loader, n_fit, keep, seed, border, out, meta["fingerprint"])
    Xes, yes, wes = _load_numpy(fit_path, feats, "early_stop", n_es, 1.0, seed)
    from catboost import Pool
    es_pool = Pool(Xes, yes, weight=wes, feature_names=feats)

    params = dict(PARAMS)
    if pool.is_quantized():
        params.pop("border_count")   # fixed by the quantised pool (254, same value)
    params.update(iterations=iterations, task_type=task_type)
    if task_type == "GPU":
        params["devices"] = "0"
    runs, p_es, models = [], [], []
    with GpuSampler() as gpu:
        for s in seeds:
            t = time.time()
            train_dir = out / "gbm_catboost_info" / f"seed{s}"
            train_dir.mkdir(parents=True, exist_ok=True)
            m = cb.CatBoostClassifier(**params, random_seed=s, train_dir=str(train_dir), metric_period=50, verbose=500,
                                      thread_count=-1)
            with stage(f"CatBoost {task_type} seed {s}"):
                m.fit(pool, eval_set=es_pool, use_best_model=True)
            f_model = out / f"gbm_model_{s}.cbm"
            models.append((m, f_model))
            p = m.predict_proba(Xes, thread_count=-1)[:, 1]
            p_es.append(p)
            imp = m.get_feature_importance(type="PredictionValuesChange")
            best = m.get_best_score().get("validation", {})
            runs.append({"seed": s, "file": f_model.name, "trees": int(m.tree_count_), "best_iteration": int(m.get_best_iteration()),
                         "best_es_logloss": best.get("Logloss"), "seconds": round(time.time() - t, 1),
                         "importance": {f: round(float(v), 4) for f, v in zip(feats, imp)}})
            log(f"seed {s}: {m.tree_count_} trees (best iteration {m.get_best_iteration()}), early-stop logloss "
                f"{best.get('Logloss')}, {fmt_secs(time.time() - t)}; peak RSS {rss_gb():.2f} GB")
    p_ens = np.mean(p_es, axis=0)
    _sanity_gate(p_es, yes, wes, [r["seed"] for r in runs])
    for m, f_model in models:
        m.save_model(str(f_model))
    ece_es = weighted_ece(p_ens, yes, wes)
    cal = None
    if ece_es > ECE_MAX:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_ens, yes, sample_weight=wes)
        cal = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(), "fitted_on": "early_stop holdout"}
        log(f"early-stop ECE {ece_es:.4f} > {ECE_MAX}: isotonic calibration fitted on the early-stop holdout")
    imp_mean = {f: round(float(np.mean([r["importance"][f] for r in runs])), 4) for f in feats}
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "features": feats, "seeds": list(seeds), "models": [r["file"] for r in runs],
        "params": {**params, "border_count": border}, "loader": loader_used, "load_seconds": round(load_s, 1),
        "fit_share_kept": keep, "n_fit_rows": int(pool.num_row()), "n_early_stop_rows": int(n_es),
        "fit_fingerprint": meta["fingerprint"], "runs": runs, "importance_mean": dict(sorted(imp_mean.items(), key=lambda kv: -kv[1])),
        "early_stop": {"ece_weighted": ece_es, "ece_single_seed": [weighted_ece(p, yes, wes) for p in p_es]},
        "calibration": cal, "gpu": gpu.summary(), "peak_rss_gb": rss_gb(), "seconds": round(time.time() - t0, 1),
    }
    save_json(manifest, out / "gbm_model.json")
    top = list(manifest["importance_mean"].items())[:15]
    log(f"fit done in {fmt_secs(time.time() - t0)}: {len(runs)} seeds, trees {[r['trees'] for r in runs]}, early-stop ECE {ece_es:.4f}, "
        f"GPU {manifest['gpu']}, peak RSS {manifest['peak_rss_gb']:.2f} GB\n  top importances: {top}")
    return manifest


# ---------------------------------------------------------------- predict (WP-3)
def _load_models(man: dict, art: Path):
    import catboost as cb
    models = []
    for f in man["models"]:
        m = cb.CatBoostClassifier()
        m.load_model(str(ml_dir(art, "train") / f))
        models.append(m)
    return models


def _entity_frame(rows: pl.DataFrame, rule: dict | None) -> pl.DataFrame:
    """Per-entity uncertainty summary of an entity-complete frame (s1_id, country, p_gbm, c_indic[, label])."""
    g = rows.with_columns(pl.min_horizontal(pl.col("p_gbm"), 1 - pl.col("p_gbm")).alias("_m"))
    aggs = [pl.col("country").first(), pl.len().alias("n_cands"), pl.col("_m").sum().alias("u"), pl.col("p_gbm").max().alias("max_p"),
            ((pl.col("p_gbm") > 0.05) & (pl.col("p_gbm") < 0.95)).sum().alias("n_band"), (pl.col("c_indic") > 0).any().alias("has_indic")]
    if "label" in rows.columns:
        aggs.append(pl.col("label").cast(pl.Int32).sum().alias("n_pos"))
    if rule is not None:
        keep = pl.col("p_gbm") >= rule["t"]
        if rule.get("m"):
            keep &= pl.col("p_gbm").rank("ordinal", descending=True).over("s1_id") <= rule["m"]
        if rule.get("r"):
            keep &= pl.col("p_gbm") >= rule["r"] * pl.col("p_gbm").max().over("s1_id")
        g = g.with_columns(keep.alias("_keep"))
        aggs += [(pl.col("p_gbm") >= rule["t"]).sum().alias("n_over_t"), pl.col("_keep").sum().alias("n_pred")]
    return g.group_by("s1_id").agg(aggs)


def predict(artifacts_dir, split: str, *, resume: bool = True, chunk_rows: int = 1_000_000) -> Path:
    t0 = time.time()
    art = Path(artifacts_dir)
    man = model_manifest(art)
    feats = man["features"]
    fams = tuple(man["fit_fingerprint"]["families"])
    models = _load_models(man, art)
    src = "train" if split == "valid" else "test"
    out = ml_dir(art, src)
    parts_dir = out / "gbm_pred_parts" / split
    parts_dir.mkdir(parents=True, exist_ok=True)
    stamp = {"models": man["created"], "features": feats}
    stamp_f = parts_dir / "_stamp.json"
    if not resume or not stamp_f.exists() or load_json(stamp_f) != stamp:
        for f in parts_dir.glob("*.parquet"):
            f.unlink()
        save_json(stamp, stamp_f)
    seeds = man["seeds"]
    timings: dict[str, float] = {}
    n_rows = 0
    for part in pbar(candidate_part_files(art / src), desc=f"gbm predict {split}", unit="part"):
        rows_f, samp_f = parts_dir / part.name, parts_dir / f"{part.stem}.sample.parquet"
        if rows_f.exists():
            continue
        t_part = time.time()
        rows, samples = [], []
        for _, fr in iter_feature_frames(art, split, families=fams, chunk_rows=chunk_rows, timings=timings, parts=[part]):
            X = fr.select(feats).to_numpy().astype(np.float32, copy=False)
            ps = [m.predict_proba(X, thread_count=-1)[:, 1].astype(np.float32) for m in models]
            p = apply_calibration(np.mean(ps, axis=0).astype(np.float32), man.get("calibration"))
            keys = fr.select("s1_id", "cand_id", "country", pl.col("c_indic").cast(pl.Int8), *(["label"] if split == "valid" else []))
            cols = [pl.Series("p_gbm", p)]
            if split == "valid":
                cols += [pl.Series(f"p_s{s}", q) for s, q in zip(seeds, ps)]
            rows.append(keys.with_columns(cols))
            hit = pair_hash(fr["s1_id"], fr["cand_id"], 42) % np.uint64(FEATSAMPLE_MOD[split]) == 0
            samples.append(fr.filter(pl.Series(hit)).select("country", *feats).with_columns(pl.Series("p_gbm", p[hit])))
        if not rows:
            pl.DataFrame(schema={"s1_id": pl.String}).write_parquet(rows_f)
            continue
        df = pl.concat(rows)
        n_rows += df.height
        pl.concat(samples).write_parquet(samp_f)
        df.write_parquet(rows_f.with_suffix(".tmp"))
        os.replace(rows_f.with_suffix(".tmp"), rows_f)
        log(f"[{split} {part.name}] {df.height:,} rows in {time.time() - t_part:.0f}s; peak RSS {rss_gb():.2f} GB")
    return finalize(art, split, timings=timings, seconds=time.time() - t0)


def _row_files(parts_dir: Path) -> list[Path]:
    return sorted(f for f in parts_dir.glob("*.parquet") if not f.name.endswith(".sample.parquet")
                  and pl.read_parquet_schema(f).get("p_gbm") is not None)


def finalize(art: Path, split: str, *, timings: dict | None = None, seconds: float | None = None) -> Path:
    """Merge the per-part predictions, tune (valid) or load (test) the rule, write the entity file."""
    src = "train" if split == "valid" else "test"
    out = ml_dir(art, src)
    parts_dir = out / "gbm_pred_parts" / split
    files = _row_files(parts_dir)
    rows_path = out / f"gbm_{split}.parquet"
    keep_cols = None if split == "valid" else ["s1_id", "cand_id", "p_gbm"]
    lf = pl.scan_parquet(files)
    (lf if keep_cols is None else lf.select(keep_cols)).sink_parquet(rows_path.with_suffix(".tmp"), compression="zstd")
    os.replace(rows_path.with_suffix(".tmp"), rows_path)
    samples = sorted(parts_dir.glob("*.sample.parquet"))
    if samples:
        pl.scan_parquet(samples).sink_parquet(out / f"gbm_featsample_{split}.parquet")
    rule_f = ml_dir(art, "train") / "gbm_rule.json"
    if split == "valid":
        rule = tune_rule(art)
    else:
        if not rule_f.exists():
            raise SystemExit(f"{rule_f} missing: run `gbm predict --split valid` first (the rule is tuned on ml_valid)")
        rule = load_json(rule_f)["rule"]
    ent = pl.concat([_entity_frame(pl.read_parquet(f, columns=["s1_id", "country", "p_gbm", "c_indic"] +
                                                   (["label"] if split == "valid" else [])), rule) for f in files])
    ent = ent.with_columns(pl.col("u").rank("ordinal", descending=True).cast(pl.Int32).sub(1).alias("u_rank"))
    force = pl.lit(False)
    if split == "test":
        check = france_check(ent)
        save_json(check, out / "gbm_france_check.json")
        if check.get("route_all_france"):
            force = pl.col("country") == "France"
        log(f"France check: {json.dumps({k: v for k, v in check.items() if k != 'by_country'})}")
    ent = (ent.with_columns(force.alias("force_route"))
              .sort(["force_route", "u", "s1_id"], descending=[True, True, False])
              .with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("route_rank")))
    ent_path = out / f"gbm_entity_{split}.parquet"
    ent.write_parquet(ent_path)
    n_rows = pl.scan_parquet(rows_path).select(pl.len()).collect().item()
    log(f"{split}: {n_rows:,} candidate rows -> {rows_path}; {ent.height:,} entities -> {ent_path} (rule {rule})"
        + (f"; {fmt_secs(seconds)}" if seconds else ""))
    if timings:
        log("feature seconds by family: " + json.dumps({k: round(v, 1) for k, v in sorted(timings.items(), key=lambda kv: -kv[1])}))
    return rows_path


# ---------------------------------------------------------------- rule, slices, France (WP-4)
def valid_arrays(art: Path, pcols=("p_gbm",)) -> dict:
    """ml_valid rows sorted by entity with block offsets and n_truth (positives + blocking misses)."""
    from .gbm_harness import valid_blocks
    v = pl.read_parquet(ml_dir(art, "train") / "gbm_valid.parquet").sort(["s1_id", "cand_id"])
    off, n_truth = valid_blocks(v, art)
    ent = v.group_by("s1_id", maintain_order=True).agg(pl.col("country").first(), (pl.col("c_indic") > 0).any().alias("has_indic"))
    return {"v": v, "off": off, "n_truth": n_truth, "y": v["label"].to_numpy().astype(np.int64), "ent": ent,
            **{c: v[c].to_numpy().astype(np.float64) for c in v.columns if c.startswith("p_")}}


def tune_rule(art: Path) -> dict:
    from .decide import search
    a = valid_arrays(art)
    t = time.time()
    rule, f = search(a["p_gbm"], a["y"], a["off"], a["n_truth"])
    save_json({"rule": rule, "f05_ml_valid": f, "n_entities": int(len(a["off"]) - 1), "tuned_on": "all ml_valid",
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ml_dir(art, "train") / "gbm_rule.json")
    log(f"decision rule on all of ml_valid ({len(a['off']) - 1:,} entities): {rule} -> F0.5 {f:.4f} ({time.time() - t:.0f}s)")
    return rule


def france_check(ent: pl.DataFrame) -> dict:
    """Section 3.3.6: France has no training rows and no labels, so compare its test-prediction statistics with the
    US / India test slices (France shares is_india = 0 with the US). Route all of France to the LLM when its
    decision-set statistics are clearly off the labelled countries': empty-prediction share differs by > 5 points
    from the US, the KS distance of max_p vs the US is > 0.15, or its mean uncertainty u is > 2x the US's."""
    by = {}
    for c in ent["country"].unique().sort().to_list():
        e = ent.filter(pl.col("country") == c)
        mp = e["max_p"].to_numpy()
        by[c] = {"entities": e.height, "max_p_quantiles": {str(q): float(np.quantile(mp, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                 "share_empty_pred": float((e["n_pred"] == 0).mean()) if "n_pred" in e.columns else None,
                 "mean_n_pred": float(e["n_pred"].mean()) if "n_pred" in e.columns else None,
                 "mean_u": float(e["u"].mean()), "share_band": float((e["n_band"] > 0).mean()), "mean_n_cands": float(e["n_cands"].mean())}
    out = {"by_country": by, "has_france": "France" in by, "route_all_france": False, "flags": {}}
    if "France" in by and "US" in by:
        fr, us = ent.filter(pl.col("country") == "France"), ent.filter(pl.col("country") == "US")
        grid = np.linspace(0, 1, 201)
        cdf = lambda x: np.searchsorted(np.sort(x), grid, side="right") / len(x)
        ks = float(np.max(np.abs(cdf(fr["max_p"].to_numpy()) - cdf(us["max_p"].to_numpy()))))
        flags = {"empty_share_gap": abs(by["France"]["share_empty_pred"] - by["US"]["share_empty_pred"]) > 0.05,
                 "ks_max_p_vs_us": ks > 0.15, "u_ratio_vs_us": by["France"]["mean_u"] > 2 * by["US"]["mean_u"]}
        out.update(ks_max_p_vs_us=ks, flags=flags, route_all_france=bool(any(flags.values())))
    return out


def _slices(pred, y, off, n_truth, ent: pl.DataFrame, p_max: np.ndarray) -> dict:
    from .llm.metrics import block_f05
    f = block_f05(pred, y, off, n_truth)
    npred = np.add.reduceat(pred.astype(np.int64), off[:-1])
    out = {"f05": float(f.mean()), "n_entities": int(len(f))}
    for c in ent["country"].unique().sort().to_list():
        m = (ent["country"] == c).to_numpy()
        out[f"f05_{c}"] = float(f[m].mean())
    hi = ent["has_indic"].to_numpy()
    out["f05_indic_cand"], out["n_indic_cand"] = (float(f[hi].mean()) if hi.any() else None), int(hi.sum())
    hard = (p_max >= 0.05) & (p_max <= 0.95)
    out["f05_hard"], out["n_hard"] = (float(f[hard].mean()) if hard.any() else None), int(hard.sum())
    single = n_truth == 0
    out["singleton_accuracy"], out["singleton_share"] = float((npred[single] == 0).mean()), float(single.mean())
    return out


def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float | None:
    """Population stability index of b against a: decile bins of a; for a concentrated feature (a few values hold
    most of a, so the deciles collapse) the categories are a's most frequent values plus one bucket for the rest."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 100 or len(b) < 100:
        return None
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) >= 3:
        ca = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a)
        cb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b)
    else:
        vals, cnt = np.unique(a, return_counts=True)
        top = vals[np.argsort(-cnt, kind="stable")[:bins]]
        share = lambda z: np.r_[[np.mean(z == v) for v in top], np.mean(~np.isin(z, top))]
        ca, cb = share(a), share(b)
    ca, cb = np.maximum(ca, 1e-4), np.maximum(cb, 1e-4)
    return float(np.sum((cb - ca) * np.log(cb / ca)))


def report(artifacts_dir) -> dict:
    from .decide import search
    from .llm.metrics import apply_rule, bce_with_logits, block_f05, ece, f05_from_counts, roc_auc
    art = Path(artifacts_dir)
    man = model_manifest(art)
    a = valid_arrays(art)
    y, off, n_truth, p = a["y"], a["off"], a["n_truth"], a["p_gbm"]
    rule = load_json(ml_dir(art, "train") / "gbm_rule.json")["rule"]
    pred = apply_rule(p, off, rule["t"], rule["m"] or None, rule["r"])
    p_max = np.maximum.reduceat(p, off[:-1])
    m: dict = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": rule, "n_valid_rows": int(len(p)),
               "ml_valid": _slices(pred, y, off, n_truth, a["ent"], p_max)}
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    m["calibration"] = {"ece": ece(p, y), "logloss": bce_with_logits(np.log(pc / (1 - pc)), y), "auc": roc_auc(p, y),
                        "isotonic_applied": man.get("calibration") is not None, "early_stop_ece": man["early_stop"]["ece_weighted"]}
    edges = np.linspace(0, 1, 11)
    idx = np.minimum((p * 10).astype(int), 9)
    m["reliability"] = [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int((idx == i).sum()),
                         "mean_p": float(p[idx == i].mean()) if (idx == i).any() else None,
                         "rate": float(y[idx == i].mean()) if (idx == i).any() else None} for i in range(10)]
    per_seed = {}
    for s in man["seeds"]:
        q = a.get(f"p_s{s}")
        if q is None:
            continue
        r, f = search(q, y, off, n_truth)
        qc = np.clip(q, 1e-7, 1 - 1e-7)
        per_seed[str(s)] = {"f05": f, "rule": r, "logloss": bce_with_logits(np.log(qc / (1 - qc)), y), "auc": roc_auc(q, y), "ece": ece(q, y)}
    m["per_seed"] = per_seed
    m["ensemble_f05"] = m["ml_valid"]["f05"]
    # loss decomposition: blocking misses (irreducible here) vs decision errors; uncertainty concentration
    f_model = block_f05(pred, y, off, n_truth)
    pos = np.add.reduceat(y, off[:-1])
    f_oracle = f05_from_counts(pos, pos, n_truth)
    loss, loss_block = 1 - f_model.mean(), 1 - f_oracle.mean()
    m["loss"] = {"total": float(loss), "blocking_misses": float(loss_block), "decisions": float(loss - loss_block),
                 "blocking_share": float(loss_block / max(loss, 1e-12)), "ceiling_perfect_decisions": float(f_oracle.mean()),
                 "entities_with_decision_error": float(((f_oracle - f_model) > 1e-9).mean())}
    seg = np.repeat(np.arange(len(off) - 1), np.diff(off))
    unc = np.bincount(seg, weights=np.minimum(p, 1 - p))
    gain = f_oracle - f_model
    order = np.argsort(-unc, kind="stable")
    m["uncertainty_routing"] = [{"top_share": q, "entities": int(len(order) * q),
                                 "fixable_loss_share": float(gain[order[:int(len(order) * q)]].sum() / max(gain.sum(), 1e-12)),
                                 "f05_if_perfect_stage_b": float((f_model.sum() + gain[order[:int(len(order) * q)]].sum()) / len(gain))}
                                for q in U_TABLE]
    m["importance_top20"] = dict(list(man["importance_mean"].items())[:20])
    # train (ml_valid rows) vs test feature distributions (section 7: degree depends on part size)
    ml = ml_dir(art, "train")
    fv, ft = ml / "gbm_featsample_valid.parquet", ml_dir(art, "test") / "gbm_featsample_test.parquet"
    if fv.exists() and ft.exists():
        sv, st = pl.read_parquet(fv), pl.read_parquet(ft)
        st_lab = st.filter(pl.col("country") != "France")
        drift = {}
        for c in man["features"] + ["p_gbm"]:
            av, at, al = (x[c].cast(pl.Float64).to_numpy() for x in (sv, st, st_lab))
            q = lambda z: [float(v) for v in np.nanquantile(z, [0.1, 0.5, 0.9, 0.99])] if np.isfinite(z).any() else None
            drift[c] = {"psi_test": _psi(av, at), "psi_test_us_india": _psi(av, al), "q_valid": q(av), "q_test": q(at)}
        m["drift"] = drift
        m["drift_flagged"] = {c: d["psi_test_us_india"] for c, d in drift.items() if (d["psi_test_us_india"] or 0) > 0.25}
    # test entity file and France check
    et = ml_dir(art, "test") / "gbm_entity_test.parquet"
    if et.exists():
        tq = ml_dir(art, "test") / "gbm_text_q.parquet"
        n_s1 = (pl.scan_parquet(tq).select(pl.len()).collect().item() if tq.exists()
                else load_frame(art / "test" / "normalized_source1.pkl").height)
        e = pl.read_parquet(et)
        m["test"] = {"entity_rows": e.height, "test_s1_entities": n_s1, "complete": e.height == n_s1,
                     "no_candidates": n_s1 - e.height, "n_force_route": int(e["force_route"].sum())}
        fc = ml_dir(art, "test") / "gbm_france_check.json"
        if fc.exists():
            m["france"] = load_json(fc)
    save_json(m, ml / "gbm_metrics.json")
    _print_report(m)
    (ml / "gbm_report.md").write_text(_report_markdown(m), encoding="utf-8")
    log(f"markdown report -> {ml / 'gbm_report.md'}")
    return m


def _report_markdown(m: dict) -> str:
    """The WP-4 report as markdown (pasted into CATBOOST_RESULTS.md / CATBOOST_PLAN.md after the full run)."""
    v, c, L = m["ml_valid"], m["calibration"], m["loss"]
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    rows = [("macro F0.5 (ensemble of 3 seeds)", f(v["f05"])),
            ("per seed", ", ".join(f"{s}: {d['f05']:.4f}" for s, d in m["per_seed"].items())),
            *[(f"F0.5 {k[4:]}", f(x)) for k, x in v.items() if k.startswith("f05_") and k not in ("f05_indic_cand", "f05_hard")],
            (f"F0.5 Indic-candidate entities (n = {v['n_indic_cand']:,})", f(v["f05_indic_cand"])),
            (f"F0.5 hard: max_p in [0.05, 0.95] (n = {v['n_hard']:,})", f(v["f05_hard"])),
            (f"singleton accuracy (share {v['singleton_share']:.3f})", f(v["singleton_accuracy"])),
            ("ECE / logloss / AUC", f"{c['ece']:.4f} / {c['logloss']:.4f} / {c['auc']:.5f}" + (" (isotonic applied)" if c["isotonic_applied"] else "")),
            ("loss = blocking misses + decisions", f"{L['total']:.4f} = {L['blocking_misses']:.4f} ({L['blocking_share']:.0%}) + {L['decisions']:.4f}; "
                                                  f"errors on {L['entities_with_decision_error']:.1%} of entities; ceiling {L['ceiling_perfect_decisions']:.4f}")]
    out = [f"# Stage-A CatBoost report ({m['created']})", "",
           f"Rule on all of ml_valid ({v['n_entities']:,} entities, {m['n_valid_rows']:,} candidate rows): "
           f"t = {m['rule']['t']}, m = {m['rule']['m']}, r = {m['rule']['r']}", "",
           "| metric (ml_valid) | value |", "|---|---|", *[f"| {k} | {x} |" for k, x in rows], "",
           "Reliability (bin: n, mean p, rate): " + "; ".join(f"{r['bin']}: {r['n']:,}, {r['mean_p']:.3f}, {r['rate']:.3f}"
                                                              for r in m["reliability"] if r["n"]), "",
           "| top share by u | entities | share of fixable loss | F0.5 with a perfect stage B |", "|---|---|---|---|",
           *[f"| {r['top_share']:.0%} | {r['entities']:,} | {r['fixable_loss_share']:.1%} | {r['f05_if_perfect_stage_b']:.4f} |"
             for r in m["uncertainty_routing"]], ""]
    if "drift" in m:
        out += ["Train/test drift (PSI of ml_valid vs test US+India rows): " +
                (", ".join(f"{k} {x:.2f}" for k, x in sorted(m["drift_flagged"].items(), key=lambda kv: -kv[1])) if m["drift_flagged"] else "none > 0.25"), ""]
    if "test" in m:
        t = m["test"]
        out += [f"Test entity file: {t['entity_rows']:,} rows of {t['test_s1_entities']:,} test S1 entities "
                f"({'complete' if t['complete'] else str(t['no_candidates']) + ' without candidates'}); forced routing {t['n_force_route']:,}.", ""]
    if "france" in m:
        fr = m["france"]
        out += [f"France check: route_all_france = {fr['route_all_france']}, flags {fr['flags']}, KS(max_p) vs US = {f(fr.get('ks_max_p_vs_us'), 3)}", "",
                "| country | entities | empty prediction | mean n_pred | mean u | share with a band candidate | median max_p |", "|---|---|---|---|---|---|---|",
                *[f"| {cn} | {d['entities']:,} | {d['share_empty_pred']:.3f} | {d['mean_n_pred']:.2f} | {d['mean_u']:.3f} | {d['share_band']:.3f} | {d['max_p_quantiles']['0.5']:.3f} |"
                  for cn, d in fr["by_country"].items()], ""]
    out += ["Top importances: " + ", ".join(f"{k} {x:.2f}" for k, x in list(m["importance_top20"].items())[:15]), ""]
    return "\n".join(out)


def _print_report(m: dict) -> None:
    v = m["ml_valid"]
    L = [f"stage-A CatBoost on all of ml_valid ({v['n_entities']:,} entities, {m['n_valid_rows']:,} rows), rule {m['rule']}",
         f"  macro F0.5 {v['f05']:.4f} | " + " | ".join(f"{k[4:]} {x:.4f}" for k, x in v.items() if k.startswith("f05_") and x is not None),
         f"  singleton accuracy {v['singleton_accuracy']:.4f} (share {v['singleton_share']:.3f}); Indic-candidate entities {v['n_indic_cand']:,}; hard {v['n_hard']:,}",
         f"  calibration: ECE {m['calibration']['ece']:.4f}, logloss {m['calibration']['logloss']:.4f}, AUC {m['calibration']['auc']:.5f}, "
         f"isotonic {m['calibration']['isotonic_applied']}",
         "  per seed: " + ", ".join(f"{s}: {d['f05']:.4f}" for s, d in m["per_seed"].items()) + f" | ensemble {m['ensemble_f05']:.4f}",
         f"  loss {m['loss']['total']:.4f} = blocking misses {m['loss']['blocking_misses']:.4f} ({m['loss']['blocking_share']:.0%}) + "
         f"decisions {m['loss']['decisions']:.4f}; errors on {m['loss']['entities_with_decision_error']:.1%} of entities",
         "  reliability: " + " ".join(f"[{r['bin']} n={r['n']} p={r['mean_p'] or 0:.3f} y={r['rate'] or 0:.3f}]" for r in m["reliability"] if r["n"]),
         "  uncertainty routing (top share by u -> share of fixable loss, F0.5 with a perfect stage B):"]
    L += [f"    top {r['top_share']:>4.0%} ({r['entities']:>7,}): {r['fixable_loss_share']:6.1%} -> {r['f05_if_perfect_stage_b']:.4f}"
          for r in m["uncertainty_routing"]]
    if "drift" in m:
        for c in ("A_cand_degree", "A_degree_rel", "p_gbm"):
            if c in m["drift"]:
                d = m["drift"][c]
                L.append(f"  drift {c}: PSI valid->test {d['psi_test']}, ->test US+India {d['psi_test_us_india']}; "
                         f"quantiles valid {d['q_valid']} test {d['q_test']}")
        L.append(f"  features with PSI > 0.25 (valid vs test US+India): {m['drift_flagged'] or 'none'}")
    if "test" in m:
        L.append(f"  test entity file: {m['test']['entity_rows']:,} rows of {m['test']['test_s1_entities']:,} test S1 entities; "
                 f"forced routing {m['test']['n_force_route']:,}")
    if "france" in m:
        L.append(f"  France check: route_all_france={m['france']['route_all_france']} flags {m['france']['flags']} "
                 f"KS {m['france'].get('ks_max_p_vs_us')}")
        for c, d in m["france"]["by_country"].items():
            L.append(f"    {c:<7} {d['entities']:>8,} entities: empty-pred {d['share_empty_pred']:.3f}, mean n_pred {d['mean_n_pred']:.2f}, "
                     f"mean u {d['mean_u']:.3f}, band {d['share_band']:.3f}, median max_p {d['max_p_quantiles']['0.5']:.3f}")
    log("\n" + "\n".join(L))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--artifacts-dir", default="artifacts")
    f.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    f.add_argument("--loader", default="auto", choices=["auto", "numpy", "file"])
    f.add_argument("--max-fit-gb", type=float, default=6.5, help="float32 fit-matrix budget (section 1)")
    f.add_argument("--iterations", type=int, default=PARAMS["iterations"])
    f.add_argument("--cpu", action="store_true", help="CPU training (default GPU)")
    f.add_argument("--refit", action="store_true", help="train again although gbm_model.json matches the fit matrix")
    p = sub.add_parser("predict")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--split", required=True, choices=["valid", "test"])
    p.add_argument("--chunk-rows", type=int, default=1_000_000)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--finalize-only", action="store_true", help="re-merge existing part predictions (e.g. after a new rule)")
    r = sub.add_parser("report")
    r.add_argument("--artifacts-dir", default="artifacts")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        with stage("gbm fit"):
            fit(a.artifacts_dir, seeds=tuple(int(s) for s in a.seeds.split(",")), loader=a.loader, max_fit_gb=a.max_fit_gb,
                iterations=a.iterations, task_type="CPU" if a.cpu else "GPU", refit=a.refit)
    elif a.cmd == "predict":
        with stage(f"gbm predict {a.split}"):
            if a.finalize_only:
                finalize(Path(a.artifacts_dir), a.split)
            else:
                predict(a.artifacts_dir, a.split, resume=not a.no_resume, chunk_rows=a.chunk_rows)
    else:
        with stage("gbm report"):
            report(a.artifacts_dir)


if __name__ == "__main__":
    main()
