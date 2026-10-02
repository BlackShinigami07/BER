"""Metrics over flat per-candidate arrays grouped into blocks by offsets (numpy only).

Layout used everywhere: `p`/`labels` are flat over all candidates, `off` (len n_blocks + 1) delimits the blocks,
`n_truth` is the entity's true-match count (including matches blocking or the top-K selection missed), so the
per-block F0.5 below is the exact challenge metric for that S1 entity given predictions inside its block.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def bce_with_logits(logits, labels) -> float:
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))))


def roc_auc(scores, labels) -> float:
    """Rank-based AUC with average ranks for ties; nan when one class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(p, labels, bins: int = 15) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / p.size)


def f05_from_counts(tp, npred, ntruth) -> np.ndarray:
    """Per-entity F0.5 with the singleton rule (empty truth + empty prediction = 1.0)."""
    tp = np.asarray(tp, dtype=np.float64)
    npred = np.asarray(npred, dtype=np.float64)
    ntruth = np.asarray(ntruth, dtype=np.float64)
    f = np.where((npred == 0) & (ntruth == 0), 1.0, 0.0)
    ok = tp > 0
    prec = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    rec = np.divide(tp, ntruth, out=np.zeros_like(tp), where=ntruth > 0)
    f[ok] = 1.25 * prec[ok] * rec[ok] / (0.25 * prec[ok] + rec[ok])
    return f


def segment_ids(off) -> np.ndarray:
    off = np.asarray(off, dtype=np.int64)
    return np.repeat(np.arange(len(off) - 1), np.diff(off))


def _check_off(off, n) -> np.ndarray:
    off = np.asarray(off, dtype=np.int64)
    if off[0] != 0 or off[-1] != n or np.any(np.diff(off) <= 0):
        raise ValueError("offsets must start at 0, end at len(values) and delimit non-empty blocks")
    return off


def block_counts(pred, labels, off):
    off = _check_off(off, len(pred))
    pred = np.asarray(pred, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int64)
    return np.add.reduceat(pred & lab, off[:-1]), np.add.reduceat(pred, off[:-1])


def block_f05(pred, labels, off, n_truth) -> np.ndarray:
    tp, npred = block_counts(pred, labels, off)
    return f05_from_counts(tp, npred, n_truth)


def apply_rule(p, off, t: float, m: int | None = None, r: float = 0.0) -> np.ndarray:
    """Decision rule: keep candidates with p >= t, at most the top-m per block, and only those >= r x block max."""
    p = np.asarray(p, dtype=np.float64)
    off = _check_off(off, len(p))
    keep = p >= t
    if r > 0:
        mx = np.maximum.reduceat(p, off[:-1])
        keep &= p >= r * mx[segment_ids(off)]
    if m:
        seg = segment_ids(off)
        order = np.lexsort((-p, seg))
        rank = np.empty(len(p), dtype=np.int64)
        rank[order] = np.arange(len(p)) - off[seg[order]]
        keep &= rank < m
    return keep


def best_threshold(p, labels, off, n_truth, grid=None) -> tuple[float, float]:
    """Best plain threshold for macro F0.5 over the blocks (the eval-time summary used during training)."""
    grid = np.round(np.arange(0.05, 0.991, 0.01), 2) if grid is None else grid
    best = (0.5, -1.0)
    for t in grid:
        f = float(block_f05(np.asarray(p) >= t, labels, off, n_truth).mean())
        if f > best[1]:
            best = (float(t), f)
    return best


def fit_temperature(logits, labels, lo: float = -3.0, hi: float = 3.0, iters: int = 80) -> float:
    """One-parameter temperature T minimising BCE(logits / T) (T19); golden-section search on log T."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.size == 0 or y.min() == y.max():
        return 1.0

    def loss(u):
        return bce_with_logits(x / np.exp(u), y)

    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return float(np.exp((a + b) / 2))


def summarize_blocks(logits, labels, off, n_truth, *, p_base=None, masks: dict | None = None, temperature: float = 1.0) -> dict:
    """Eval summary for training logs and reports: BCE, AUC, ECE, best-threshold macro F0.5 overall and per slice,
    and the same F0.5 for a baseline probability (stage A) on the very same blocks."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    off = np.asarray(off, dtype=np.int64)
    n_truth = np.asarray(n_truth)
    p = sigmoid(logits / temperature)
    t, f = best_threshold(p, labels, off, n_truth)
    out = {"n_blocks": int(len(off) - 1), "n_cands": int(len(p)), "bce": bce_with_logits(logits / temperature, labels),
           "auc": roc_auc(logits, labels), "ece": ece(p, labels), "f05": f, "f05_threshold": t}
    if p_base is not None:
        tb, fb = best_threshold(np.asarray(p_base, dtype=np.float64), labels, off, n_truth)
        out.update(f05_base=fb, f05_base_threshold=tb, auc_base=roc_auc(p_base, labels))
    for name, bmask in (masks or {}).items():
        bmask = np.asarray(bmask, dtype=bool)
        if not bmask.any():
            continue
        fb = block_f05(p >= t, labels, off, n_truth)
        out[f"f05_{name}"] = float(fb[bmask].mean())
        if p_base is not None:
            fbb = block_f05(np.asarray(p_base) >= out["f05_base_threshold"], labels, off, n_truth)
            out[f"f05_base_{name}"] = float(fbb[bmask].mean())
    return out
