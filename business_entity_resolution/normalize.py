"""Text normalisation for names and addresses (plan D2). No stemming, no lemmatisation, no external data.

Pipeline for every string: NFKC -> unidecode (accents stripped, Indic scripts transliterated) -> lowercase
-> web/junk tails removed -> '&' -> ' and ' -> periods/apostrophes deleted -> everything else non-alnum -> space.
Names additionally lose leading honorifics and legal-suffix tokens (canonical + learned corrupted variants).
Addresses lose null-like components, '#', leading zeros; ordinal words become ordinals; street-type
abbreviations are expanded; the state/city alias table (learned from GT) adds a canonical state token.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from unidecode import unidecode

# ------------------------------------------------------------------ vocabularies (from the EDA noise catalog)
HONORIFICS = {"mr", "mrs", "ms", "dr", "smt", "shri", "shrii", "sri", "messrs", "prof", "m/s", "ms."}
CANONICAL_SUFFIXES = {
    # en / us / in
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "plc", "lp", "llp",
    "pc", "pllc", "dba", "pvt", "private", "opc", "public",
    # fr
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "ets", "etablissements", "cie", "scp", "selarl", "sel",
    "ei", "eirl", "gie",
}
# Corrupted suffix forms observed in the data (the learner in aliases.py extends this set per split).
SEED_CORRUPTED_SUFFIXES = {"li", "limittedd", "limittett", "limirrrrdd", "elelpii", "privatelimited", "pvtltd", "ltdd"}
SUFFIX_PREFIX_RULES = ("limi", "limm", "limt", "priv", "elelp", "corpor", "incorp")
NAME_JUNK_TOKENS = {"www", "http", "https", "com", "c0m", "null", "the"}
ADDR_JUNK_TOKENS = {"null", "na", "none", "nil", "n", "hno", "no", "h"}
ADDR_WORD_MAP = {"nr": "near", "opp": "opposite", "opposite": "opposite", "bh": "behind"}
STREET_ABBR_LAST = {  # applied only to the last token of an address component (street-type suffixes)
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "blvd": "boulevard",
    "bd": "boulevard", "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "pl": "place", "sq": "square", "ter": "terrace", "trl": "trail", "pky": "parkway", "expy": "expressway",
    "mkt": "market", "twp": "township", "cdp": "", "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
}
STREET_ABBR_FR_LEAD = {  # french street types come first ("63 R. DE DIEPPE"); applied when first or after a number
    "r": "rue", "av": "avenue", "bd": "boulevard", "all": "allee", "imp": "impasse", "pl": "place", "ch": "chemin",
    "rte": "route", "sq": "square", "crs": "cours", "qu": "quai", "res": "residence", "lot": "lotissement",
}
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th",
    "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th", "fortieth": "40th",
    "fiftieth": "50th", "sixtieth": "60th", "seventieth": "70th", "eightieth": "80th", "ninetieth": "90th",
}
INDIC_RE = re.compile(r"[ऀ-෿]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

_PIPE_TAIL = re.compile(r"\s*\|.*$")                      # "| www.shivshakti.com"
_NUM_TAIL = re.compile(r"\s*[-–]\s*\d{6,}\s*$")           # "RQ Global - 4430525197"
_URL = re.compile(r"\b(?:https?://|www\.)\S+")
_TLD = re.compile(r"\.(?:com|c0m|net|org|in|fr|co|io|biz|info)\b")
_APOS = re.compile(r"[.'’‘`]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"^\d+$")
_ORD_NUM = re.compile(r"^(\d+)(st|nd|rd|th)$")


def basic_clean(s: str) -> str:
    """Language/script agnostic cleaning to a lowercase ascii token string."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = unidecode(s)
    s = s.lower()
    s = _PIPE_TAIL.sub("", s)
    s = _NUM_TAIL.sub("", s)
    s = _URL.sub(lambda m: _TLD.sub("", m.group(0).split("//")[-1].replace("www.", "")), s)
    s = _TLD.sub("", s)
    s = s.replace("&", " and ").replace("+", " plus ")
    s = _APOS.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def script_class(s: str) -> str:
    if INDIC_RE.search(s):
        return "indic"
    if NON_ASCII_RE.search(s):
        return "other"
    return "latin"


# ------------------------------------------------------------------ names
@dataclass
class NameNormalizer:
    suffixes: set[str] = field(default_factory=lambda: set(CANONICAL_SUFFIXES) | set(SEED_CORRUPTED_SUFFIXES))
    honorifics: set[str] = field(default_factory=lambda: set(HONORIFICS))

    def is_suffix(self, tok: str) -> bool:
        return tok in self.suffixes or tok.startswith(SUFFIX_PREFIX_RULES)

    def __call__(self, raw: str) -> tuple[str, str]:
        """Return (name_full, name_core) normalised strings."""
        s = re.sub(r"\bm/s\b\.?", " ", raw, flags=re.I)  # "M/s" honorific before punctuation is stripped
        full_toks = basic_clean(s).split()
        toks = list(full_toks)
        while toks and toks[0] in self.honorifics:
            toks = toks[1:]
        core = [t for t in toks if not self.is_suffix(t) and t not in NAME_JUNK_TOKENS]
        if not core:  # name consisted only of suffix/junk tokens: fall back to the full form
            core = [t for t in full_toks if t not in NAME_JUNK_TOKENS] or full_toks
        return " ".join(full_toks), " ".join(core)


# ------------------------------------------------------------------ addresses
def clean_component(c: str) -> str:
    return basic_clean(c)


def split_components(raw: str) -> list[str]:
    """Comma-separated address components, cleaned, null-like ones dropped."""
    out = []
    for c in raw.split(","):
        cc = clean_component(c)
        if not cc or cc in ADDR_JUNK_TOKENS:
            continue
        out.append(cc)
    return out


def _norm_token(tok: str) -> str:
    if _NUM.match(tok):
        return tok.lstrip("0") or "0"
    m = _ORD_NUM.match(tok)
    if m:
        return (m.group(1).lstrip("0") or "0") + m.group(2)
    return ORDINAL_WORDS.get(tok, tok)


def normalize_component_tokens(comp: str, is_state_alias: bool = False) -> list[str]:
    """Tokens of one component with numeric/ordinal/abbreviation normalisation."""
    toks = comp.split()
    if is_state_alias and len(toks) == 1:
        return toks
    out: list[str] = []
    n = len(toks)
    for i, t in enumerate(toks):
        t = _norm_token(t)
        if t in ADDR_JUNK_TOKENS:
            continue
        if i == n - 1 and t in STREET_ABBR_LAST:
            t = STREET_ABBR_LAST[t]
        elif (i == 0 or _NUM.match(toks[i - 1] or "")) and t in STREET_ABBR_FR_LEAD:
            t = STREET_ABBR_FR_LEAD[t]
        else:
            t = ADDR_WORD_MAP.get(t, t)
        if t:
            out.append(t)
    return out


@dataclass
class AddressNormalizer:
    """Builds the final address token string; `alias` maps normalised components to a canonical state."""
    alias: dict[str, str] = field(default_factory=dict)

    def state_of(self, comps: list[str]) -> str | None:
        for c in reversed(comps):
            s = self.alias.get(c)
            if s is not None:
                return s
        return None

    def __call__(self, comps: list[str]) -> tuple[str, str | None]:
        """Return (addr_norm, state_norm)."""
        state = self.state_of(comps) if self.alias else None
        toks: list[str] = []
        for c in comps:
            toks.extend(normalize_component_tokens(c, is_state_alias=(c in self.alias)))
        if state and state not in toks:
            toks.append(state)
        return " ".join(toks), state


def components_to_string(comps: list[str]) -> str:
    return "|".join(comps)


def string_to_components(s: str) -> list[str]:
    return [c for c in s.split("|") if c] if s else []
