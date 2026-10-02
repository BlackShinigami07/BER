"""Pre-tokenised blocks (the WP-B parquet contract), token-budget batch planning and collation.

A block file row = one S1 entity: `input_ids` (the S1 line then its candidate lines, BOS first), `marker_pos` (index
of each candidate line's final " ans" token), `s1_len`, `cand_ids`, `labels` (train/dev/valid), `p_gbm`, plus
per-entity flags. `blocks_meta.json` next to the files carries the tokenizer contract (pad/marker ids and the token
ids of every "\\n[k]" line prefix) so candidate lines can be re-ordered at train time without a tokenizer.

Line k (0-based) of a block spans ids[start_k : marker_pos[k] + 1] with start_0 = s1_len and
start_k = marker_pos[k-1] + 1; it is prefix_ids[k] ("\\n[k+1]") followed by the candidate body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from .runtime import list_parquet, load_json, log
except ImportError:  # SageMaker source_dir: modules are imported as top-level scripts
    from runtime import list_parquet, load_json, log

META_NAME = "blocks_meta.json"
LIST_COLS = {"input_ids": np.int32, "marker_pos": np.int32, "labels": np.int8, "p_gbm": np.float32}
TRAIN_COLS = ("input_ids", "marker_pos", "s1_len", "labels", "n_tokens")
EVAL_COLS = TRAIN_COLS + ("p_gbm", "n_truth", "country", "has_indic", "is_hard", "routed", "s1_id")
SCORE_COLS = ("s1_id", "input_ids", "marker_pos", "s1_len", "cand_ids", "n_tokens", "p_gbm")


@dataclass
class BlockMeta:
    tokenizer: str
    template_version: int
    pad_id: int
    bos_id: int
    marker_id: int
    prefix_ids: list[list[int]]
    max_tokens: int = 800
    k: int = 12
    k_max: int = 16
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "BlockMeta":
        p = Path(path)
        f = p if p.is_file() and p.name == META_NAME else None
        if f is None:
            base = p if p.is_dir() else p.parent
            hits = [base / META_NAME] if (base / META_NAME).exists() else sorted(base.rglob(META_NAME))
            if not hits:
                raise FileNotFoundError(f"{META_NAME} not found under {base} (written by ml/blocks.py next to the parquet files)")
            f = hits[0]
        d = load_json(f)
        known = {k: d[k] for k in ("tokenizer", "template_version", "pad_id", "bos_id", "marker_id", "prefix_ids", "max_tokens", "k", "k_max") if k in d}
        return cls(**known, extra={k: v for k, v in d.items() if k not in known})

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("tokenizer", "template_version", "pad_id", "bos_id", "marker_id", "prefix_ids", "max_tokens", "k", "k_max")}
        return {**self.extra, **d}


def _min_stat(pf, column: str, default):
    """Minimum of a flat column from parquet row-group statistics (no data read); `default` when unavailable."""
    md, lo = pf.metadata, None
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            if cc.path_in_schema == column:
                st = cc.statistics
                if st is None or not st.has_min_max:
                    return default
                lo = st.min if lo is None else min(lo, st.min)
    return default if lo is None else lo


def _flatten_list_column(col, dtype):
    import pyarrow as pa
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    off = np.asarray(arr.offsets, dtype=np.int64)
    vals = arr.flatten().to_numpy(zero_copy_only=False)
    if dtype is not None:
        vals = vals.astype(dtype, copy=False)
    return vals, off - off[0]


class BlockStore:
    """All blocks of a subset held as flat numpy arrays (a few hundred MB for 250k blocks)."""

    def __init__(self, cols: dict, meta: BlockMeta, files: list[Path]):
        self.meta = meta
        self.files = files
        self.ids, self.ids_off = cols.pop("input_ids")
        self.mpos, self.m_off = cols.pop("marker_pos")
        self.labels = cols.pop("labels", (None, None))[0]
        self.p_gbm = cols.pop("p_gbm", (None, None))[0]
        cand = cols.pop("cand_ids", None)
        self.cand_ids = cand[0] if cand is not None else None
        self.s1_len = cols.pop("s1_len").astype(np.int64)
        self.n_tokens = np.diff(self.ids_off).astype(np.int64)
        self.cols = cols  # scalar per-block columns (s1_id, country, flags, sample_rank, n_truth, ...)
        self._prefix = [np.asarray(p, dtype=np.int32) for p in meta.prefix_ids]
        self._plen = np.array([len(p) for p in meta.prefix_ids], dtype=np.int64)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path, *, columns=EVAL_COLS, n_blocks: int = 0, shard: tuple[int, int] | None = None,
             routed_only: bool = False, limit: int = 0, meta: BlockMeta | None = None, quiet: bool = False,
             keep_ids=None) -> "BlockStore":
        """Load one parquet file or a directory of shards.

        n_blocks > 0 keeps rows with sample_rank < n_blocks (a quota-preserving prefix of the WP-B sample: this is
        how the training-set size is chosen at launch time); shard=(rank, world) keeps rows i % world == rank of
        every file; routed_only keeps rows flagged by the inference router; keep_ids (a pyarrow array of s1_ids)
        keeps only those entities (the CatBoost uncertainty routing, CATBOOST_PLAN.md section 5); limit caps the
        rows per call.
        """
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        files = list_parquet(path)
        meta = meta or BlockMeta.load(Path(path) if Path(path).is_dir() else Path(path).parent)
        want = set(columns) | {"input_ids", "marker_pos", "s1_len"}
        parts: dict[str, list] = {}
        used, n_rows = [], 0
        for f in files:
            pf = pq.ParquetFile(f)
            names = set(pf.schema_arrow.names)
            if n_blocks and "sample_rank" in names and _min_stat(pf, "sample_rank", default=0) >= n_blocks:
                continue  # the whole file is beyond the requested prefix: never read it (saves S3 streaming)
            cols = [c for c in want | {"sample_rank", "routed"} | ({"s1_id"} if keep_ids is not None else set()) if c in names]
            t = pq.read_table(f, columns=cols)
            if keep_ids is not None:
                t = t.filter(pc.is_in(t["s1_id"], value_set=keep_ids))
            if n_blocks and "sample_rank" in t.column_names:
                t = t.filter(pc.less(t["sample_rank"], n_blocks))
            if routed_only and "routed" in t.column_names:
                t = t.filter(t["routed"])
            if shard is not None and shard[1] > 1:
                t = t.take(np.nonzero(np.arange(t.num_rows) % shard[1] == shard[0])[0])
            if limit and n_rows + t.num_rows > limit:
                t = t.slice(0, max(0, limit - n_rows))
            if t.num_rows == 0:
                continue
            n_rows += t.num_rows
            used.append(f)
            for c in t.column_names:
                if c not in want and c not in ("sample_rank", "routed"):
                    continue
                if c in LIST_COLS or c == "cand_ids":
                    parts.setdefault(c, []).append(_flatten_list_column(t[c], LIST_COLS.get(c)))
                else:
                    parts.setdefault(c, []).append(t[c].to_numpy(zero_copy_only=False))
            if limit and n_rows >= limit:
                break
        if n_rows == 0:
            raise ValueError(f"no blocks selected from {path} (n_blocks={n_blocks}, shard={shard}, routed_only={routed_only}, "
                             f"keep_ids={None if keep_ids is None else len(keep_ids)})")
        cols = {}
        for c, chunks in parts.items():
            if c in LIST_COLS or c == "cand_ids":
                vals = np.concatenate([v for v, _ in chunks])
                offs, base = [np.zeros(1, dtype=np.int64)], 0
                for v, o in chunks:
                    offs.append(o[1:] + base)
                    base += len(v)
                cols[c] = (vals, np.concatenate(offs))
            else:
                cols[c] = np.concatenate(chunks)
        store = cls(cols, meta, used)
        store._check()
        if not quiet:
            log(f"loaded {len(store):,} blocks ({int(store.n_tokens.sum()):,} tokens, {len(store.mpos):,} candidates) "
                f"from {len(used)} file(s) under {path}")
        return store

    def _check(self) -> None:
        if not np.array_equal(np.diff(self.m_off) > 0, np.ones(len(self), dtype=bool)):
            raise ValueError("every block needs at least one candidate marker")
        sample = np.arange(min(len(self), 2000))
        for i in sample:
            a, b = self.ids_off[i], self.ids_off[i + 1]
            mp = self.mpos[self.m_off[i]:self.m_off[i + 1]]
            if mp[-1] != b - a - 1 or np.any(np.diff(mp) <= 0) or np.any(self.ids[a + mp] != self.meta.marker_id):
                raise ValueError(f"block {i}: marker positions do not match the '{self.meta.marker_id}' marker tokens")

    # ------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.ids_off) - 1

    def n_cands(self, i: int) -> int:
        return int(self.m_off[i + 1] - self.m_off[i])

    def col(self, name: str):
        return self.cols.get(name)

    def labels_of(self, i: int):
        return None if self.labels is None else self.labels[self.m_off[i]:self.m_off[i + 1]]

    def block(self, i: int, perm: np.ndarray | None = None):
        """(input_ids, marker_pos, labels) of block i; `perm` re-orders the candidate lines (perm[j] = original line
        placed at position j) and renumbers them "[1]".."[k]" in the new order, exactly as WP-B would have written it."""
        a, b = self.ids_off[i], self.ids_off[i + 1]
        ids = self.ids[a:b]
        mpos = self.mpos[self.m_off[i]:self.m_off[i + 1]]
        lab = self.labels_of(i)
        if perm is None:
            return ids, mpos, lab
        s1 = int(self.s1_len[i])
        starts = np.concatenate([[s1], mpos[:-1] + 1])
        pieces, new_mpos, pos = [ids[:s1]], np.empty(len(mpos), dtype=np.int32), s1
        for j, k in enumerate(perm):
            body = ids[starts[k] + self._plen[k]: mpos[k] + 1]
            pieces += [self._prefix[j], body]
            pos += self._plen[j] + len(body)
            new_mpos[j] = pos - 1
        return np.concatenate(pieces), new_mpos, (None if lab is None else lab[perm])


# ---------------------------------------------------------------- batch planning (T9)
def padded_len(n: int, multiple: int = 8) -> int:
    return int(-(-n // multiple) * multiple)


def plan_batches(n_tokens: np.ndarray, token_budget: int, *, seed: int = 0, epoch: int = 0, shuffle: bool = True,
                 jitter: float = 16.0, max_blocks: int = 256, pad_multiple: int = 8) -> list[np.ndarray]:
    """Length-grouped token-budget micro-batches: sort blocks by length (+ a little seeded jitter so batches change
    between epochs), cut greedily so that (#blocks x padded max length) <= token_budget, then shuffle batch order."""
    n_tokens = np.asarray(n_tokens)
    rng = np.random.default_rng([seed, epoch, 7])
    key = n_tokens + (rng.uniform(0, jitter, len(n_tokens)) if shuffle and jitter > 0 else 0)
    order = np.argsort(key, kind="stable")
    batches, start, cur_max = [], 0, 0
    for j, i in enumerate(order):
        m = max(cur_max, padded_len(int(n_tokens[i]), pad_multiple))
        if j > start and (m * (j - start + 1) > token_budget or j - start >= max_blocks):
            batches.append(order[start:j])
            start, m = j, padded_len(int(n_tokens[i]), pad_multiple)
        cur_max = m
    if len(order) > start:
        batches.append(order[start:])
    if shuffle:
        batches = [batches[k] for k in rng.permutation(len(batches))]
    return batches


def rank_batches(batches: list[np.ndarray], rank: int, world: int, multiple_of: int = 1) -> list[np.ndarray]:
    """Batches of one rank: round-robin, truncated so every rank has the same count (a multiple of the grad
    accumulation). Equal counts per rank are what keeps DDP from hanging with variable-size token batches."""
    n = (len(batches) // world) // multiple_of * multiple_of
    return batches[rank::world][:n]


def block_perm(seed: int, epoch: int, block: int, k: int) -> np.ndarray:
    """Deterministic per-(epoch, block) candidate order: independent of worker/rank assignment, so resume is exact."""
    return np.random.default_rng([seed, epoch, int(block)]).permutation(k)


def collate(store: BlockStore, idx, *, perms: list | None = None, pad_multiple: int = 8, with_labels: bool = True) -> dict:
    """Right-padded batch: input_ids / attention_mask (B x T), one entry per candidate marker (M) with its batch row,
    position and label. Numpy only (runs in DataLoader workers); tensors are created at the end."""
    import torch
    items = [store.block(int(i), None if perms is None else perms[j]) for j, i in enumerate(idx)]
    T = padded_len(max(len(x[0]) for x in items), pad_multiple)
    B = len(items)
    ids = np.full((B, T), store.meta.pad_id, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.int64)
    n_m = np.array([len(x[1]) for x in items], dtype=np.int64)
    mbi = np.repeat(np.arange(B), n_m)
    mpos = np.concatenate([x[1] for x in items]).astype(np.int64)
    for r, (x, _, _) in enumerate(items):
        ids[r, :len(x)] = x
        mask[r, :len(x)] = 1
    out = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask),
           "marker_batch_idx": torch.from_numpy(mbi), "marker_pos": torch.from_numpy(mpos),
           "n_markers": torch.from_numpy(n_m), "block_index": torch.as_tensor(np.asarray(idx, dtype=np.int64)),
           "n_real_tokens": int(mask.sum()), "n_padded_tokens": int(B * T)}
    if with_labels and store.labels is not None:
        out["labels"] = torch.from_numpy(np.concatenate([x[2] for x in items]).astype(np.float32))
    return out


class BatchDataset:
    """Map-style dataset whose items are whole micro-batches (DataLoader(batch_size=None)); candidate order is
    shuffled per (epoch, block) when `shuffle_cands` (training), canonical or reversed otherwise."""

    def __init__(self, store: BlockStore, batches: list[np.ndarray], *, shuffle_cands: bool = False, seed: int = 0,
                 epoch: int = 0, reverse: bool = False, pad_multiple: int = 8, with_labels: bool = True):
        self.store, self.batches = store, batches
        self.shuffle_cands, self.seed, self.epoch, self.reverse = shuffle_cands, seed, epoch, reverse
        self.pad_multiple, self.with_labels = pad_multiple, with_labels

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, j: int) -> dict:
        idx = self.batches[j]
        perms = None
        if self.shuffle_cands:
            perms = [block_perm(self.seed, self.epoch, i, self.store.n_cands(int(i))) for i in idx]
        elif self.reverse:
            perms = [np.arange(self.store.n_cands(int(i)))[::-1] for i in idx]
        out = collate(self.store, idx, perms=perms, pad_multiple=self.pad_multiple, with_labels=self.with_labels)
        if perms is not None:
            import torch
            out["perm"] = torch.from_numpy(np.concatenate(perms).astype(np.int64))
        return out
