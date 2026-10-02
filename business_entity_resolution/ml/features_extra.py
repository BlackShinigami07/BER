"""Label-free "orthogonal" pair-feature families for the CatBoost stage-A model (CATBOOST_PLAN.md sections 3.2, 3.3).

Every family runs in one of two places, so that train, valid and test compute each feature identically:

* part level (`Family.part_fn`): needs every row of one candidate part (a state shard), e.g. the candidate graph.
  It runs on the whole part - all S1 entities, all folds, all rows - before any entity subsample or row filter,
  which is exactly what the test split sees.
* entity level (`Family.fn`): needs every candidate of each S1 entity (the feature builder cuts parts into
  entity-complete chunks) and runs after the base features of ml/features.py.

Nothing here reads `label` or `fold`; ids are used only as grouping keys, never as values. Undefined comparisons
(an empty side) are null, not 0 or 1, because "no evidence" and "evidence of a mismatch" are different things.

`PRODUCTION_FAMILIES` is the model schema: section 3.2's measured families A-E plus the WP-1 families that passed
the paired test (CATBOOST_RESULTS.md). The other registered families stay available to the ablation harness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from ..io import load_pkl

F32 = pl.Float32
PIN_RE = r"\b\d{5,6}\b"          # 5-digit US zip / 6-digit Indian PIN on the raw address


@dataclass
class ExtraLookups:
    """Split-level lookup tables the families need (all label-free, see load_extra_lookups)."""
    idf: pl.DataFrame | None = None     # country, tok, idf        (blocking's word TF-IDF vectorizer of this split)
    oov: pl.DataFrame | None = None     # country, oov             (IDF of a token the vectorizer never saw)
    alias: pl.DataFrame | None = None   # country, comp            (city / area address components)


def load_extra_lookups(split_dir: str | Path) -> ExtraLookups:
    """IDF per country from the split's own word vectorizer (fitted on that split's candidate pool, like blocking),
    and the city/area components of the learned alias table (train/aliases.pkl; the test split reuses train's)."""
    split_dir = Path(split_dir)
    out = ExtraLookups()
    vec_f = split_dir / "vectorizers.pkl"
    if vec_f.exists():
        idf_parts, oov = [], {}
        for country, vectors in load_pkl(vec_f).get("vectorizers", {}).items():
            word = vectors.get("word") if isinstance(vectors, dict) else None
            if word is None or not hasattr(word, "idf_"):
                continue
            idf = np.asarray(word.idf_, dtype=np.float32)
            idf_parts.append(pl.DataFrame({"country": str(country), "tok": word.get_feature_names_out().tolist(), "idf": idf}))
            stop = sorted(getattr(word, "stop_words_", None) or ())       # dropped by max_df: the most common tokens
            if stop:
                idf_parts.append(pl.DataFrame({"country": str(country), "tok": stop, "idf": np.full(len(stop), idf.min(), np.float32)}))
            # smooth idf = ln((1+n)/(1+df)) + 1: df=1 is the max, an unseen token (df=0) is max + ln 2 (approximately)
            oov[str(country)] = float(idf.max() + math.log(2.0))
        if idf_parts:
            out.idf = pl.concat(idf_parts).unique(["country", "tok"], keep="first")
            out.oov = pl.DataFrame({"country": list(oov), "oov": np.asarray(list(oov.values()), dtype=np.float32)})
    for alias_f in (split_dir / "aliases.pkl", split_dir.parent / "train" / "aliases.pkl"):
        if alias_f.exists():
            rows = [(str(c), comp) for c, t in load_pkl(alias_f).items() for comp in t.get("alias", {})
                    if comp not in set(t.get("states", ()))]
            out.alias = pl.DataFrame(rows, schema={"country": pl.String, "comp": pl.String}, orient="row").unique()
            break
    return out


# ---------------------------------------------------------------- small helpers
def cpdist(scorer, a: np.ndarray, b: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Element-wise similarity of a[i] vs b[i] in [0, 1] (rapidfuzz.process.cpdist on all cores)."""
    from rapidfuzz import process
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / np.float32(scale)


def _toks(col: str) -> pl.Expr:
    """Whitespace tokens (duplicates kept)."""
    return pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _tokset(col: str) -> pl.Expr:
    return _toks(col).list.unique()


def _comps(col: str) -> pl.Expr:
    return pl.col(col).fill_null("").str.split("|").list.eval(pl.element().filter(pl.element() != ""))


def _jacc(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a.list.set_intersection(b).list.len() / pl.max_horizontal(a.list.set_union(b).list.len(), pl.lit(1))).cast(F32)


def _masked(values: np.ndarray, ok: np.ndarray) -> pl.Series:
    return pl.Series(np.where(ok, values, np.float32(np.nan)).astype(np.float32)).fill_nan(None)


def _np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].fill_null("").to_numpy()


# ---------------------------------------------------------------- 3.2 A: candidate graph (part level)
def part_A_graph(part: pl.DataFrame, ctx=None) -> pl.DataFrame:
    """A generic / distractor candidate is retrieved by many S1 entities of its shard; a true match is usually
    mutual-best. Degree also normalised by the part's entity count (train and test parts differ in size)."""
    n_ent = max(1, part["s1_id"].n_unique())
    return (part.with_columns(pl.col("s1_id").n_unique().over("cand_id").cast(F32).alias("A_cand_degree"),
                              pl.col("rrf_score").rank("min", descending=True).over("cand_id").cast(F32).alias("A_reverse_rank"))
                .with_columns((pl.col("A_cand_degree") / n_ent).cast(F32).alias("A_degree_rel")))


# ---------------------------------------------------------------- 3.2 B: address structure
def fam_B_addr_struct(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Address components (city / area identity that token soup blurs) and raw postal codes."""
    qc, cc = _comps("q_comps"), _comps("c_comps")
    qpin = pl.col("q_raw").fill_null("").str.extract_all(PIN_RE).list.unique()
    cpin = pl.col("c_raw").fill_null("").str.extract_all(PIN_RE).list.unique()
    have_comp = (qc.list.len() > 0) & (cc.list.len() > 0)
    have_pin = (qpin.list.len() > 0) & (cpin.list.len() > 0)
    return df.with_columns(
        pl.when(have_comp).then(_jacc(qc.list.unique(), cc.list.unique())).otherwise(None).cast(F32).alias("B_comp_jacc"),
        pl.when(have_comp).then(qc.list.last() == cc.list.last()).otherwise(None).cast(F32).alias("B_last_comp_equal"),
        pl.when(have_pin).then(_jacc(qpin, cpin)).otherwise(None).cast(F32).alias("B_pin_jacc"),
    )


# ---------------------------------------------------------------- 3.2 C: name structure
def fam_C_name_struct(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Containment (abbreviations, truncations), legal-suffix mass of the candidate, exact core-token Jaccard."""
    q, c = pl.col("q_core").fill_null(""), pl.col("c_core").fill_null("")
    both = (q != "") & (c != "")
    return df.with_columns(
        pl.when(both).then(c.str.contains(q, literal=True)).otherwise(None).cast(F32).alias("C_q_in_c"),
        pl.when(both).then(q.str.contains(c, literal=True)).otherwise(None).cast(F32).alias("C_c_in_q"),
        (pl.col("c_full").fill_null("").str.len_chars().cast(pl.Int32) - c.str.len_chars().cast(pl.Int32))
          .clip(0, None).cast(F32).alias("C_c_suffix_len"),
        pl.when(both).then(_jacc(_tokset("q_core"), _tokset("c_core"))).otherwise(None).cast(F32).alias("C_tok_jacc"),
    )


# ---------------------------------------------------------------- 3.2 D: entity context (needs A)
def fam_D_entity_ctx(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Ambiguity of the whole block: near-duplicate candidates, best name / address evidence, candidate popularity."""
    g = "s1_id"
    return df.with_columns(
        (pl.col("core_tset") >= 0.9).sum().over(g).cast(F32).alias("D_n_near_dup"),
        pl.col("core_tset").max().over(g).cast(F32).alias("D_best_core"),
        pl.col("addr_tset").max().over(g).cast(F32).alias("D_best_addr"),
        (pl.col("core_tset") - pl.col("core_tset").mean().over(g)).cast(F32).alias("D_core_vs_mean"),
        pl.col("A_cand_degree").mean().over(g).cast(F32).alias("D_mean_degree"),
    )


# ---------------------------------------------------------------- 3.2 E: IDF of shared core tokens
def _mass(frame: pl.DataFrame, key: str, col: str, lk: ExtraLookups) -> pl.DataFrame:
    e = (frame.select(key, "country", _tokset(col).alias("tok")).explode("tok", empty_as_null=False).drop_nulls("tok")
              .join(lk.idf, on=["country", "tok"], how="left").join(lk.oov, on="country", how="left"))
    return e.group_by(key, "country").agg(pl.coalesce("idf", "oov").sum().alias("m"))


def fam_E_idf(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Sharing a rare token ("aditya") means more than sharing a common one ("consultants"): IDF mass of the shared
    core tokens over each side's mass, and the rarest shared token. Vocabulary = this split's blocking word
    vectorizer per country (fitted on the split's own pool); tokens it never saw get the unseen-token IDF."""
    cols = ("E_idf_frac_q", "E_idf_frac_c", "E_idf_max_shared")
    if lk is None or lk.idf is None:
        return df.with_columns([pl.lit(None, F32).alias(c) for c in cols])
    df = df.with_columns(pl.col("country").fill_null("").cast(pl.String).alias("_cty"))
    mq = _mass(df.select("s1_id", pl.col("_cty").alias("country"), "q_core").unique(["s1_id", "country"]), "s1_id", "q_core", lk)
    mc = _mass(df.select("cand_id", pl.col("_cty").alias("country"), "c_core").unique(["cand_id", "country"]), "cand_id", "c_core", lk)
    sh = (df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r"), pl.col("_cty").alias("country"),
                    _tokset("q_core").list.set_intersection(_tokset("c_core")).alias("tok"))
            .explode("tok", empty_as_null=False).drop_nulls("tok")
            .join(lk.idf, on=["country", "tok"], how="left").join(lk.oov, on="country", how="left")
            .with_columns(pl.coalesce("idf", "oov").alias("w"))
            .group_by("_r").agg(pl.col("w").sum().alias("ms"), pl.col("w").max().alias("mx")))
    out = (df.with_row_index("_r")
             .join(mq.rename({"country": "_cty", "m": "_mq"}), on=["s1_id", "_cty"], how="left", maintain_order="left")
             .join(mc.rename({"country": "_cty", "m": "_mc"}), on=["cand_id", "_cty"], how="left", maintain_order="left")
             .join(sh, on="_r", how="left", maintain_order="left"))
    both = (pl.col("q_core").fill_null("") != "") & (pl.col("c_core").fill_null("") != "")
    ms = pl.col("ms").fill_null(0.0)
    return out.with_columns(
        pl.when(both).then(ms / pl.max_horizontal(pl.col("_mq"), pl.lit(1e-6))).otherwise(None).cast(F32).alias("E_idf_frac_q"),
        pl.when(both).then(ms / pl.max_horizontal(pl.col("_mc"), pl.lit(1e-6))).otherwise(None).cast(F32).alias("E_idf_frac_c"),
        pl.when(both).then(pl.col("mx").fill_null(0.0)).otherwise(None).cast(F32).alias("E_idf_max_shared"),
    ).drop("_r", "_cty", "_mq", "_mc", "ms", "mx")


# ---------------------------------------------------------------- 3.3.1 transitivity / co-candidate agreement
def fam_F1_transitivity(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Similarity of candidate c to the entity's best other candidates (by the blocking fusion order: rrf_rank,
    then rrf_score, then id): o1 = the top candidate (the second one for the top row itself), o2 = the next one.
    "c and the best candidate are duplicates of each other" is evidence for both (S2/S3 hold several copies)."""
    top = (df.select("s1_id", "cand_id", "rrf_rank", "rrf_score", "c_core", "c_norm")
             .sort(["s1_id", "rrf_rank", "rrf_score", "cand_id"], descending=[False, False, True, False])
             .group_by("s1_id", maintain_order=True)
             .agg(pl.col("cand_id").head(3).alias("_tid"), pl.col("c_core").head(3).alias("_tcore"),
                  pl.col("c_norm").head(3).alias("_tnorm")))
    d = df.select("s1_id", "cand_id", "c_core", "c_norm").join(top, on="s1_id", how="left", maintain_order="left")
    is0 = pl.col("cand_id") == pl.col("_tid").list.get(0, null_on_oob=True)
    is1 = pl.col("cand_id") == pl.col("_tid").list.get(1, null_on_oob=True)
    i1 = pl.when(is0).then(1).otherwise(0)
    i2 = pl.when(is0 | is1).then(2).otherwise(1)
    d = d.select(pl.col("c_core").fill_null(""), pl.col("c_norm").fill_null(""),
                 pl.col("_tcore").list.get(i1, null_on_oob=True).alias("o1_core"), pl.col("_tnorm").list.get(i1, null_on_oob=True).alias("o1_norm"),
                 pl.col("_tcore").list.get(i2, null_on_oob=True).alias("o2_core"), pl.col("_tnorm").list.get(i2, null_on_oob=True).alias("o2_norm"))
    from rapidfuzz import fuzz
    cc, cn = _np(d, "c_core"), _np(d, "c_norm")
    out = {}
    for o in ("o1", "o2"):
        oc, on = _np(d, f"{o}_core"), _np(d, f"{o}_norm")
        ok_c = d[f"{o}_core"].is_not_null().to_numpy() & (cc != "") & (oc != "")
        ok_a = d[f"{o}_norm"].is_not_null().to_numpy() & (cn != "") & (on != "")
        out[f"F1_{o}_core"] = _masked(cpdist(fuzz.token_set_ratio, cc, oc), ok_c)
        out[f"F1_{o}_addr"] = _masked(cpdist(fuzz.token_set_ratio, cn, on), ok_a)
    return df.with_columns([s.alias(k) for k, s in out.items()])


# ---------------------------------------------------------------- 3.3.2 graph, deeper (part + entity level; needs A)
def part_F2_graph_deep(part: pl.DataFrame, ctx) -> pl.DataFrame:
    """Part-level popularity flag (degree above the part's 90th percentile of candidate degree) and how many distinct
    candidates of the part carry the same core name (generic names such as "shree ganesh traders")."""
    n_ent = max(1, part["s1_id"].n_unique())
    cands = part.select("cand_id", "A_cand_degree").unique("cand_id")
    q90 = float(cands["A_cand_degree"].quantile(0.9, "nearest") or 0.0)
    core = ctx.pool_text(cands["cand_id"], ["c_core"])
    freq = (core.filter(pl.col("c_core").fill_null("") != "").group_by("c_core").agg(pl.len().alias("_f"))
                .join(core, on="c_core").select("cand_id", pl.col("_f").cast(F32).alias("F2_c_core_freq")))
    return (part.join(freq, on="cand_id", how="left", maintain_order="left")
                .with_columns((pl.col("A_cand_degree") > q90).cast(F32).alias("F2_is_popular"),
                              (pl.col("F2_c_core_freq") / n_ent * 1000).cast(F32).alias("F2_c_core_freq_rel")))


def fam_F2_graph_deep(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Mutual best (c is this S1's top candidate AND this S1 is c's top suitor) and the S1 side's popularity."""
    g = "s1_id"
    return df.with_columns(
        ((pl.col("rrf_rank") == pl.col("rrf_rank").min().over(g)) & (pl.col("A_reverse_rank") == 1)).cast(F32).alias("F2_mutual_best"),
        pl.col("F2_is_popular").sum().over(g).cast(F32).alias("F2_s1_n_popular"),
    )


# ---------------------------------------------------------------- 3.3.3 noise-type signatures
_CONFUSABLE = (["0", "1", "i"], ["o", "l", "l"])


def fam_F3_noise(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Signatures of the data's corruption catalogue (scripts/eda/00_noise_catalog.py): transpositions (Damerau vs
    Levenshtein), digit/letter confusions (1/l/i, 0/o), common subsequence, suffix-only differences, token counts,
    numeric share of the shared tokens. rapidfuzz has no longest common *substring*; LCSseq (subsequence) is the
    cheap stand-in, and partial_ratio (already a base feature) covers the substring alignment."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import DamerauLevenshtein, LCSseq, Levenshtein
    conf = lambda x: pl.col(x).fill_null("").str.replace_many(*_CONFUSABLE)
    d = df.select(pl.col("q_core").fill_null(""), pl.col("c_core").fill_null(""), conf("q_core").alias("qx"), conf("c_core").alias("cx"))
    qc, cc = _np(d, "q_core"), _np(d, "c_core")
    ok = (qc != "") & (cc != "")
    dl = cpdist(DamerauLevenshtein.normalized_similarity, qc, cc, 1.0)
    lv = cpdist(Levenshtein.normalized_similarity, qc, cc, 1.0)
    conf_ratio = cpdist(fuzz.ratio, _np(d, "qx"), _np(d, "cx"))
    qt, ct = _tokset("q_core"), _tokset("c_core")
    shared = qt.list.set_intersection(ct)
    n_shared = shared.list.len()
    n_num = shared.list.eval(pl.element().str.contains(r"^\d+$")).list.sum()
    return df.with_columns(
        _masked(dl - lv, ok).alias("F3_dl_gap"),
        _masked(conf_ratio - df["core_ratio"].fill_null(0).to_numpy(), ok).alias("F3_confus_gain"),
        _masked(cpdist(LCSseq.normalized_similarity, qc, cc, 1.0), ok).alias("F3_lcsseq"),
        ((pl.col("q_core") == pl.col("c_core")) & (pl.col("q_full") != pl.col("c_full"))).cast(F32).alias("F3_suffix_only"),
        (pl.col("core_ratio") - pl.col("full_ratio")).cast(F32).alias("F3_core_full_gap"),
        (_toks("q_core").list.len().cast(pl.Int32) - _toks("c_core").list.len().cast(pl.Int32)).abs().cast(F32).alias("F3_tok_count_diff"),
        pl.when(n_shared > 0).then(n_num / n_shared).otherwise(None).cast(F32).alias("F3_num_tok_share"),
    )


# ---------------------------------------------------------------- 3.3.4 phonetic / transliteration
_PHON: dict[str, tuple[str, str]] = {}


def _phonetic(s: str) -> tuple[str, str]:
    """(metaphone string, soundex string) of a core name, token by token; numeric tokens pass through."""
    hit = _PHON.get(s)
    if hit is None:
        import jellyfish
        toks = s.split()
        meta = " ".join(t if not t.isalpha() else (jellyfish.metaphone(t) or t) for t in toks)
        sdx = " ".join(jellyfish.soundex(t) for t in toks if t.isalpha())
        if len(_PHON) > 3_000_000:
            _PHON.clear()
        hit = _PHON[s] = (meta.lower(), sdx.lower())
    return hit


def _phon_frame(values: pl.Series) -> pl.DataFrame:
    u = values.fill_null("").unique()
    codes = [_phonetic(s) for s in u.to_list()]
    return pl.DataFrame({"s": u, "meta": [c[0] for c in codes], "sdx": [c[1] for c in codes]})


def fam_F4_phonetic(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Metaphone / Soundex agreement of core tokens and the consonant skeleton (transliteration keeps consonants and
    mangles vowels), plus one explicit c_indic x core_tset interaction."""
    from rapidfuzz import fuzz
    ph = _phon_frame(pl.concat([df["q_core"], df["c_core"]]))
    d = (df.select(pl.col("q_core").fill_null(""), pl.col("c_core").fill_null(""))
           .join(ph.rename({"s": "q_core", "meta": "q_meta", "sdx": "q_sdx"}), on="q_core", how="left", maintain_order="left")
           .join(ph.rename({"s": "c_core", "meta": "c_meta", "sdx": "c_sdx"}), on="c_core", how="left", maintain_order="left")
           .with_columns(pl.col("q_core").str.replace_all("[aeiou]", "").str.strip_chars().alias("q_cons"),
                         pl.col("c_core").str.replace_all("[aeiou]", "").str.strip_chars().alias("c_cons")))
    qm, cm, qs, cs = (_np(d, x) for x in ("q_meta", "c_meta", "q_cons", "c_cons"))
    ok_m, ok_s = (qm != "") & (cm != ""), (qs != "") & (cs != "")
    sdx = d.select(pl.when((pl.col("q_sdx") != "") & (pl.col("c_sdx") != ""))
                     .then(_jacc(_tokset("q_sdx"), _tokset("c_sdx"))).otherwise(None).cast(F32).alias("x"))["x"]
    return df.with_columns(
        _masked(cpdist(fuzz.token_set_ratio, qm, cm), ok_m).alias("F4_meta_tset"),
        sdx.alias("F4_sdx_jacc"),
        _masked(cpdist(fuzz.ratio, qs, cs), ok_s).alias("F4_cons_ratio"),
        _masked(cpdist(fuzz.token_set_ratio, qs, cs), ok_s).alias("F4_cons_tset"),
        (pl.col("c_indic") * pl.col("core_tset")).cast(F32).alias("F4_indic_x_core"),
    )


# ---------------------------------------------------------------- 3.3.5 address, deeper
def _cities(frame: pl.DataFrame, key: str, col: str, lk: ExtraLookups) -> pl.DataFrame:
    e = frame.select(key, "country", _comps(col).alias("comp")).explode("comp", empty_as_null=False).drop_nulls("comp")
    return e.join(lk.alias, on=["country", "comp"], how="semi").group_by(key, "country").agg(pl.col("comp").unique().alias("cities"))


def fam_F5_addr_deep(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """City/area agreement through the learned alias table (components known to belong to a state), number-sequence
    order agreement, postal code on one side only, address token-count ratio."""
    num = lambda x: pl.col(x).fill_null("").str.extract_all(r"\d+")
    ntok = lambda x: _toks(x).list.len()
    pin = lambda x: pl.col(x).fill_null("").str.contains(PIN_RE)
    out = df.with_columns(
        pl.when((num("q_norm").list.len() >= 2) & (num("c_norm").list.len() >= 2))
          .then(num("q_norm").list.get(1, null_on_oob=True) == num("c_norm").list.get(1, null_on_oob=True)).otherwise(None).cast(F32).alias("F5_second_num_equal"),
        pl.when((num("q_norm").list.len() >= 1) & (num("c_norm").list.len() >= 1))
          .then(num("q_norm") == num("c_norm")).otherwise(None).cast(F32).alias("F5_num_seq_equal"),
        (pin("q_raw") != pin("c_raw")).cast(F32).alias("F5_pin_one_side"),
        pl.when((ntok("q_norm") > 0) & (ntok("c_norm") > 0))
          .then(pl.min_horizontal(ntok("q_norm"), ntok("c_norm")) / pl.max_horizontal(ntok("q_norm"), ntok("c_norm")))
          .otherwise(None).cast(F32).alias("F5_addr_ntok_ratio"),
    )
    if lk is None or lk.alias is None:
        return out.with_columns(pl.lit(None, F32).alias("F5_city_match"), pl.lit(None, F32).alias("F5_city_jacc"))
    out = out.with_columns(pl.col("country").fill_null("").cast(pl.String).alias("_cty"))
    qcity = _cities(out.select("s1_id", pl.col("_cty").alias("country"), "q_comps").unique(["s1_id", "country"]), "s1_id", "q_comps", lk)
    ccity = _cities(out.select("cand_id", pl.col("_cty").alias("country"), "c_comps").unique(["cand_id", "country"]), "cand_id", "c_comps", lk)
    out = (out.join(qcity.rename({"country": "_cty", "cities": "_qc"}), on=["s1_id", "_cty"], how="left", maintain_order="left")
              .join(ccity.rename({"country": "_cty", "cities": "_cc"}), on=["cand_id", "_cty"], how="left", maintain_order="left"))
    both = (pl.col("_qc").list.len() > 0) & (pl.col("_cc").list.len() > 0)
    inter = pl.col("_qc").list.set_intersection(pl.col("_cc")).list.len()
    return out.with_columns(
        pl.when(both).then(inter > 0).otherwise(None).cast(F32).alias("F5_city_match"),
        pl.when(both).then(_jacc(pl.col("_qc"), pl.col("_cc"))).otherwise(None).cast(F32).alias("F5_city_jacc"),
    ).drop("_cty", "_qc", "_cc")


# ---------------------------------------------------------------- 3.3.7 target-free entity priors
def fam_F7_entity_priors(df: pl.DataFrame, lk: ExtraLookups | None = None) -> pl.DataFrame:
    """Share of the entity's candidates that come from S3, and the normalised entropy of its rrf_score distribution
    (a flat distribution = no clear winner). The count of candidates with n_blockers >= 3 is already ctx_nb3."""
    g = "s1_id"
    p = pl.col("rrf_score").cast(pl.Float64) / pl.col("rrf_score").cast(pl.Float64).sum().over(g)
    n = pl.len().over(g)
    return df.with_columns(
        (pl.col("src") == "S3").cast(F32).mean().over(g).cast(F32).alias("F7_share_s3"),
        pl.when(n > 1).then((-(p * p.log()).fill_nan(0.0)).sum().over(g) / n.cast(pl.Float64).log())
          .otherwise(None).cast(F32).alias("F7_rrf_entropy"),
    )


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class Family:
    name: str
    columns: tuple[str, ...]
    fn: Callable | None = None           # entity level: (df, lookups) -> df
    part_fn: Callable | None = None      # part level:   (part, ctx) -> part
    part_columns: tuple[str, ...] = ()   # the columns part_fn adds
    requires: tuple[str, ...] = ()
    plan: str = ""


# Explicit, in dependency order, so every ablation is reproducible and unknown names fail before an expensive run.
FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("A_graph", ("A_cand_degree", "A_reverse_rank", "A_degree_rel"), part_fn=part_A_graph,
           part_columns=("A_cand_degree", "A_reverse_rank", "A_degree_rel"), plan="3.2 A"),
    Family("B_addr_struct", ("B_comp_jacc", "B_last_comp_equal", "B_pin_jacc"), fn=fam_B_addr_struct, plan="3.2 B"),
    Family("C_name_struct", ("C_q_in_c", "C_c_in_q", "C_c_suffix_len", "C_tok_jacc"), fn=fam_C_name_struct, plan="3.2 C"),
    Family("D_entity_ctx", ("D_n_near_dup", "D_best_core", "D_best_addr", "D_core_vs_mean", "D_mean_degree"),
           fn=fam_D_entity_ctx, requires=("A_graph",), plan="3.2 D"),
    Family("E_idf", ("E_idf_frac_q", "E_idf_frac_c", "E_idf_max_shared"), fn=fam_E_idf, plan="3.2 E"),
    Family("F1_transitivity", ("F1_o1_core", "F1_o1_addr", "F1_o2_core", "F1_o2_addr"), fn=fam_F1_transitivity, plan="3.3.1"),
    Family("F2_graph_deep", ("F2_c_core_freq", "F2_c_core_freq_rel", "F2_is_popular", "F2_mutual_best", "F2_s1_n_popular"),
           fn=fam_F2_graph_deep, part_fn=part_F2_graph_deep, part_columns=("F2_c_core_freq", "F2_c_core_freq_rel", "F2_is_popular"),
           requires=("A_graph",), plan="3.3.2"),
    Family("F3_noise", ("F3_dl_gap", "F3_confus_gain", "F3_lcsseq", "F3_suffix_only", "F3_core_full_gap", "F3_tok_count_diff",
                        "F3_num_tok_share"), fn=fam_F3_noise, plan="3.3.3"),
    Family("F4_phonetic", ("F4_meta_tset", "F4_sdx_jacc", "F4_cons_ratio", "F4_cons_tset", "F4_indic_x_core"),
           fn=fam_F4_phonetic, plan="3.3.4"),
    Family("F5_addr_deep", ("F5_second_num_equal", "F5_num_seq_equal", "F5_pin_one_side", "F5_addr_ntok_ratio",
                            "F5_city_match", "F5_city_jacc"), fn=fam_F5_addr_deep, plan="3.3.5"),
    Family("F7_entity_priors", ("F7_share_s3", "F7_rrf_entropy"), fn=fam_F7_entity_priors, plan="3.3.7"),
)}
MEASURED_FAMILIES = ("A_graph", "B_addr_struct", "C_name_struct", "D_entity_ctx", "E_idf")
EXPLORE_FAMILIES = tuple(n for n in FAMILIES if n not in MEASURED_FAMILIES)
# The model schema: 3.2's measured families + the WP-1 families that passed the paired test (CATBOOST_RESULTS.md).
PRODUCTION_FAMILIES = MEASURED_FAMILIES


def resolve(families) -> tuple[str, ...]:
    """Requested families plus their requirements, in registry order."""
    names = MEASURED_FAMILIES if families is None else tuple(families)
    unknown = [n for n in names if n not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown feature families {unknown}; choose from {list(FAMILIES)}")
    want = set(names)
    for n in names:
        want.update(FAMILIES[n].requires)
    return tuple(n for n in FAMILIES if n in want)


def family_columns(families) -> list[str]:
    return [c for n in resolve(families) for c in FAMILIES[n].columns]


def part_columns(families) -> list[str]:
    return [c for n in resolve(families) for c in FAMILIES[n].part_columns]


def add_part_families(part: pl.DataFrame, ctx, families, timings: dict | None = None) -> pl.DataFrame:
    """Part-level families on ALL rows of one candidate part (label-free); families already present are kept."""
    import time
    for n in resolve(families):
        fam = FAMILIES[n]
        if fam.part_fn is None or all(c in part.columns for c in fam.part_columns):
            continue
        t = time.perf_counter()
        part = fam.part_fn(part, ctx)
        if timings is not None:
            timings[n] = timings.get(n, 0.0) + time.perf_counter() - t
    return part


def add_families(df: pl.DataFrame, lookups: ExtraLookups | None, families, timings: dict | None = None) -> pl.DataFrame:
    """Entity-level families on an entity-complete frame that already has the base features and part-level columns."""
    import time
    for n in resolve(families):
        fam = FAMILIES[n]
        if fam.fn is None:
            continue
        t = time.perf_counter()
        df = fam.fn(df, lookups)
        if timings is not None:
            timings[n] = timings.get(n, 0.0) + time.perf_counter() - t
    return df
