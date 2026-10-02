"""TSV readers (tab-separated, no quoting, everything as strings, empty stays ""), pkl/parquet helpers and
artifact layout.

artifacts/{split}/
    normalized_{source}.pkl            ML-facing: one pandas DataFrame per source (all countries)
    parts/normalized_{source}_{cc}.parquet   pipeline-internal per-country parts
    countries.json                     {safe_name: country}
    candidate_parts/{cc}__{shard}.parquet    pipeline-internal candidate parts (one per state shard)
    candidates_{k:02d}.pkl             ML-facing pkl parts (<= pkl_part_rows rows each) + candidates_index.json
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pandas as pd
import polars as pl

from .progress import log

READ_KW = dict(separator="\t", quote_char=None, infer_schema_length=0, empty_string_is_null=False)
SOURCES = ("source1", "source2", "source3")
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def source_path(data_dir: Path, split: str, source: str) -> Path:
    return Path(data_dir) / split / f"{split}_{source}.tsv"


def read_source(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, **READ_KW)
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {df.columns}")
    df = df.select(SOURCE_COLS).with_columns([pl.col(c).fill_null("") for c in SOURCE_COLS])
    log(f"read {path.name}: {df.height:,} rows")
    return df


def read_ground_truth(path: Path) -> pl.DataFrame:
    gt = pl.read_csv(path, **READ_KW).with_columns([pl.col(c).fill_null("") for c in ["source1_entity_id", "matched_entity_ids"]])
    log(f"read {path.name}: {gt.height:,} rows")
    return gt


def gt_to_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Explode the ground truth into (s1_id, cand_id) rows (singletons produce no rows)."""
    return (
        gt.filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"})
        .with_columns(pl.col("cand_id").str.strip_chars())
        .filter(pl.col("cand_id") != "")
    )


# ---------------------------------------------------------------- layout helpers
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "x"


def normalized_part(split_dir: Path, source: str, country: str) -> Path:
    return Path(split_dir) / "parts" / f"normalized_{source}_{safe_name(country)}.parquet"


def save_countries(split_dir: Path, countries: list[str]) -> None:
    save_json({safe_name(c): c for c in sorted(countries)}, Path(split_dir) / "countries.json")


def list_countries(split_dir: Path) -> list[str]:
    with open(Path(split_dir) / "countries.json", encoding="utf-8") as f:
        return sorted(json.load(f).values())


def candidate_parts_dir(split_dir: Path) -> Path:
    return Path(split_dir) / "candidate_parts"


def candidate_parts(split_dir: Path) -> list[Path]:
    return sorted(candidate_parts_dir(split_dir).glob("*.parquet"))


def scan_candidates(split_dir: Path) -> pl.LazyFrame:
    files = candidate_parts(split_dir)
    if not files:
        raise FileNotFoundError(f"no candidate parts under {candidate_parts_dir(split_dir)}; run the block stage first")
    return pl.scan_parquet([str(f) for f in files])


def load_candidates_pandas(split_dir: Path) -> pd.DataFrame:
    """ML-stage helper: concatenate the pkl parts into one pandas DataFrame (needs RAM for the whole table)."""
    with open(Path(split_dir) / "candidates_index.json", encoding="utf-8") as f:
        idx = json.load(f)
    return pd.concat([pd.read_pickle(Path(split_dir) / p["file"]) for p in idx["parts"]], ignore_index=True)


# ---------------------------------------------------------------- pkl helpers
def save_pkl(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pl.DataFrame):
        obj = obj.to_pandas(use_pyarrow_extension_array=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)
    log(f"saved {path} ({path.stat().st_size / 1e6:,.1f} MB)")
    return path


def load_pkl(path: Path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def load_frame(path: Path) -> pl.DataFrame:
    """Load a pickled pandas DataFrame back into polars."""
    obj = load_pkl(path)
    if isinstance(obj, pl.DataFrame):
        return obj
    if isinstance(obj, pd.DataFrame):
        return pl.from_pandas(obj)
    raise TypeError(f"{path}: expected a DataFrame, got {type(obj)}")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    log(f"saved {path}")


def load_json(path: Path):
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)
