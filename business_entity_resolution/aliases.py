"""Learn normalisation vocabularies from the data itself (plan D3). No external databases or APIs.

* corrupted legal-suffix vocabulary: frequent S2/S3 name tokens within a small edit distance of a canonical
  suffix ("limittedd", "limirrrrdd", ...), excluding tokens that are also frequent in clean S1 names.
* state vocabulary + alias table per country from ground-truth pairs: an S2/S3 address component maps to an
  S1 state when P(state | component) >= purity and the component was seen often enough. Captures codes (MH),
  transliterated native-script names ("mhaaraassttr"), typos ("keerlln") and city -> state ("houston" -> tx).
"""
from __future__ import annotations

import collections
import re

import polars as pl
from rapidfuzz.distance import Levenshtein

from .normalize import CANONICAL_SUFFIXES, SEED_CORRUPTED_SUFFIXES, string_to_components
from .progress import log, pbar

_HAS_DIGIT = re.compile(r"\d")


# ------------------------------------------------------------------ suffix vocabulary (unsupervised)
def learn_suffix_vocab(pool_tokens: collections.Counter, pool_last_tokens: collections.Counter,
                       s1_tokens: collections.Counter, min_count: int = 100, s1_ratio: float = 0.05,
                       last_ratio: float = 0.6) -> set[str]:
    """Corrupted legal-suffix tokens: frequent in S2/S3 names, mostly in the LAST position (like real suffixes),
    rare in clean S1 names, and within a small edit distance of a canonical suffix."""
    learned = set(SEED_CORRUPTED_SUFFIXES)
    long_canon = [c for c in CANONICAL_SUFFIXES if len(c) >= 5]
    mid_canon = [c for c in CANONICAL_SUFFIXES if 3 <= len(c) < 5]
    for tok, n in pool_tokens.items():
        if n < min_count or tok in CANONICAL_SUFFIXES or len(tok) < 3:
            continue
        if pool_last_tokens.get(tok, 0) < last_ratio * n:   # not predominantly a trailing token
            continue
        if s1_tokens.get(tok, 0) > s1_ratio * n:             # genuine word in clean S1 names -> not a suffix
            continue
        if any(Levenshtein.distance(tok, c) <= 2 for c in long_canon) or \
           any(Levenshtein.distance(tok, c) <= 1 for c in mid_canon):
            learned.add(tok)
    return learned


# ------------------------------------------------------------------ state / city aliases (from GT)
def learn_state_aliases(s1_comps: list[list[str]], pool_comps_by_pair: list[list[str]], s1_comps_by_pair: list[list[str]],
                        min_state_count: int = 50, min_alias_count: int = 30, purity: float = 0.95,
                        demote_ratio: float = 0.2) -> dict:
    """Returns {"states": [...], "alias": {component: state}} for one country."""
    lastc: collections.Counter = collections.Counter()
    for comps in s1_comps:
        if comps and not _HAS_DIGIT.search(comps[-1]):
            lastc[comps[-1]] += 1
    cand = {k for k, v in lastc.items() if v >= min_state_count}
    before: collections.Counter = collections.Counter()
    for comps in s1_comps:
        if len(comps) >= 2 and comps[-1] in cand:
            for x in comps[:-1]:
                if x in cand and x != comps[-1]:
                    before[x] += 1
    states = {v for v in cand if before[v] < demote_ratio * lastc[v]}

    def state_of(comps):
        for x in reversed(comps):
            if x in states:
                return x
        return None

    co: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for ca, cb in zip(s1_comps_by_pair, pool_comps_by_pair):
        sa = state_of(ca)
        if sa is None:
            continue
        for x in set(cb):
            co[x][sa] += 1
    alias = {s: s for s in states}
    for v, cnt in co.items():
        tot = sum(cnt.values())
        top, n = cnt.most_common(1)[0]
        if tot >= min_alias_count and n / tot >= purity:
            alias[v] = top
    return {"states": sorted(states), "alias": alias, "n_states": len(states), "n_alias": len(alias)}


def learn_all_state_aliases(s1: pl.DataFrame, pool: pl.DataFrame, pairs: pl.DataFrame) -> dict[str, dict]:
    """s1/pool: normalised frames with entity_id, country, addr_comps. pairs: (s1_id, cand_id)."""
    j = (pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country"), pl.col("addr_comps").alias("ca")), on="s1_id")
         .join(pool.select(pl.col("entity_id").alias("cand_id"), pl.col("addr_comps").alias("cb")), on="cand_id"))
    out: dict[str, dict] = {}
    for country in sorted(s1["country"].unique().to_list()):
        sub = j.filter(pl.col("country") == country)
        s1c = [string_to_components(x) for x in pbar(s1.filter(pl.col("country") == country)["addr_comps"].to_list(), desc=f"aliases[{country}] S1 comps")]
        ca = [string_to_components(x) for x in pbar(sub["ca"].to_list(), desc=f"aliases[{country}] pair S1")]
        cb = [string_to_components(x) for x in pbar(sub["cb"].to_list(), desc=f"aliases[{country}] pair S2/S3")]
        res = learn_state_aliases(s1c, cb, ca)
        log(f"aliases[{country}]: {res['n_states']} states, {res['n_alias']} alias entries "
            f"(sample: {list(res['alias'].items())[:6]})")
        out[country] = res
    return out
