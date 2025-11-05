#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_batch_search_box_qty_colab_optimized.py - REFACTORED
------------------------------------------------------------
Fixed critical bugs:
1. Missing doc_names/doc_codes attributes in HybridSearch.score()
2. Python 3.9 compatible type hints
3. Improved synonym replacement logic
4. Better error handling
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import pandas as _pd
import json

try:
    from rapidfuzz.distance import Levenshtein as _RFLev
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# -----------------------------------------------------------------------------
# Global caches and mappings
# -----------------------------------------------------------------------------

synonyms_map: Dict[str, str] = {}
DEFAULT_SYNONYMS_FILE = "synonyms.csv"

# Precompile frequently used regex patterns
SIZE_RE = re.compile(r"\b\d{2,3}/\d{2,3}-\d{2}\b|\b\d{2,3}/\d{2,3}\b|\b\d{2,3}-\d{2}\b")
FRONT_RE = re.compile(r"(?i)\btr(ươ|u)ớc\b|front|(?<!\w)f(?!\w)")
REAR_RE = re.compile(r"(?i)\bsau\b|rear|(?<!\w)r(?!\w)")
GENUINE_RE = re.compile(r"(?i)\bOR\b|ch(i|í)nh h(a|ă)̃ng|genuine|x(i|ị)n|jin")
BLACK_RE = re.compile(r"\bden\b|\bđen\b|\bblack\b")
SIDE_RIGHT_RE = re.compile(r"\bphai\b|\bright\b|\bp1\b|\bp 1\b")
SIDE_LEFT_RE = re.compile(r"\btrai\b|\bleft\b|\bp0\b|\bp 0\b")
NON_ALPHANUM_RE = re.compile(r"[^0-9a-zà-ỹ/.\-\s]")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALPHANUM_SYNONYMS_RE = re.compile(r"[^0-9a-z/\-\s]")

# Brand patterns for quick lookup
BRAND_PATTERNS = {
    "sh": re.compile(r"\bsh\b"),
    "pcx": re.compile(r"\bpcx\b"),
    "mode": re.compile(r"\bmode\b"),
    "lead": re.compile(r"\blead\b|\blit\b|\blít\b"),
    "jupiter": re.compile(r"\bjupiter\b|\bju\b|\bya\b|\byamaha\b"),
    "vision": re.compile(r"\bvision\b"),
    "air blade": re.compile(r"\bairblade\b|\bair blade\b|\babl\b|\bablade\b"),
    "exciter": re.compile(r"\bexciter\b|\be\b|\be135\b|\be150\b"),
    "winner": re.compile(r"\bwinner\b|\bwnr\b"),
    "sirius": re.compile(r"\bsirius\b|\bsi\b"),
    "wave": re.compile(r"\bwave\b|\brsx\b|\balpha\b"),
}


# -----------------------------------------------------------------------------
# Optimized text utilities with caching
# -----------------------------------------------------------------------------

@lru_cache(maxsize=10000)
def strip_accents(s: str) -> str:
    """Cached accent stripping."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


@lru_cache(maxsize=10000)
def normalize_text(s: str) -> str:
    """Cached text normalization."""
    s = unicodedata.normalize("NFC", s).lower()
    s = NON_ALPHANUM_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return strip_accents(s)


@lru_cache(maxsize=5000)
def _normalise_for_synonyms(text: str) -> str:
    """Cached synonym normalization."""
    text_norm = strip_accents(text).lower()
    text_norm = NON_ALPHANUM_SYNONYMS_RE.sub(" ", text_norm)
    return WHITESPACE_RE.sub(" ", text_norm).strip()


def tokens(s: str) -> List[str]:
    """Fast tokenization using cached normalize."""
    return normalize_text(s).split()


@lru_cache(maxsize=2000)
def extract_size(s: str) -> str:
    """Cached size extraction."""
    no_space = s.replace(" ", "")
    m = SIZE_RE.search(no_space) or SIZE_RE.search(s)
    return m.group(0) if m else ""


@lru_cache(maxsize=2000)
def has_front(s: str) -> bool:
    """Cached front detection."""
    return bool(FRONT_RE.search(s))


@lru_cache(maxsize=2000)
def has_rear(s: str) -> bool:
    """Cached rear detection."""
    return bool(REAR_RE.search(s))


@lru_cache(maxsize=2000)
def genuine_hint(s: str) -> bool:
    """Cached genuine detection."""
    return bool(GENUINE_RE.search(s))


def brand_hits(s: str) -> int:
    """Optimized brand detection using precompiled patterns."""
    s_norm = normalize_text(s)
    return sum(1 for pattern in BRAND_PATTERNS.values() if pattern.search(s_norm))


@lru_cache(maxsize=2000)
def detect_side(text: str, code: str = "") -> Optional[str]:
    """Cached side detection."""
    t = strip_accents(text).lower()
    c = strip_accents(code).lower()

    side_right = bool(SIDE_RIGHT_RE.search(t)) or "p1" in c
    side_left = bool(SIDE_LEFT_RE.search(t)) or "p0" in c

    if side_right and not side_left:
        return "right"
    if side_left and not side_right:
        return "left"
    return None


def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Optimized Jaccard similarity."""
    if not a_tokens or not b_tokens:
        return 0.0
    A, B = set(a_tokens), set(b_tokens)
    intersection = len(A & B)
    if intersection == 0:
        return 0.0
    return intersection / (len(A) + len(B) - intersection)


from difflib import SequenceMatcher

# Reuse SequenceMatcher instance
_seq_matcher = SequenceMatcher()

def seq_ratio(a: str, b: str) -> float:
    """Optimized sequence ratio using reused matcher."""
    _seq_matcher.set_seqs(a, b)
    return _seq_matcher.ratio()


# -----------------------------------------------------------------------------
# Synonyms handling (optimized & fixed)
# -----------------------------------------------------------------------------

def parse_synonyms_file(path: str) -> Dict[str, str]:
    """Optimized synonym parsing with early returns."""
    if not path or not os.path.isfile(path):
        return {}

    try:
        df = _pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}

    syn_map: Dict[str, str] = {}

    for _, row in df.iterrows():
        c_term = row.get("canonical_term", "") or ""
        canonical_norm = _normalise_for_synonyms(c_term)
        if not canonical_norm:
            continue

        # Process variants
        variants_field = row.get("variants_pipe", "") or ""
        variants = [v.strip() for v in str(variants_field).split("|") if v.strip()]
        variants.append(canonical_norm)

        # Process token mappings
        token_map = str(row.get("token_level_map", "") or "").strip()
        if token_map:
            for pair in token_map.split(";"):
                if "->" not in pair:
                    continue
                src, dst = pair.split("->", 1)
                src_norm = _normalise_for_synonyms(src.strip())
                dst_norm = _normalise_for_synonyms(dst.strip())
                if src_norm:
                    syn_map[src_norm] = dst_norm

        # Map variants
        for v in variants:
            v_norm = _normalise_for_synonyms(v)
            if v_norm and v_norm not in syn_map:
                syn_map[v_norm] = canonical_norm

    return syn_map


def load_synonyms(file_path: str = DEFAULT_SYNONYMS_FILE) -> None:
    """Load synonyms and clear normalization cache."""
    global synonyms_map
    synonyms_map = parse_synonyms_file(file_path)
    # Clear caches when synonyms change
    synonyms_normalize.cache_clear()
    _normalise_for_synonyms.cache_clear()


@lru_cache(maxsize=5000)
def synonyms_normalize(text: str) -> str:
    """
    Cached synonym normalization with improved replacement logic.
    Uses word boundaries to prevent substring replacement issues.
    """
    text_norm = _normalise_for_synonyms(text)
    if not synonyms_map:
        return text_norm

    # Sort by length descending for greedy replacement
    # Split into tokens to avoid substring issues
    tokens_list = text_norm.split()
    result_tokens = []

    for token in tokens_list:
        # Check if entire token matches a variant
        replaced = synonyms_map.get(token, token)
        result_tokens.append(replaced)

    return " ".join(result_tokens)


def generate_synonyms_suggestions(query: str, selected_name: str) -> List[Tuple[str, str]]:
    """Generate synonym suggestions (optimized)."""
    suggestions: List[Tuple[str, str]] = []
    q_norm = synonyms_normalize(query)
    sel_norm = synonyms_normalize(selected_name)
    q_tokens = set(tokens(q_norm))
    sel_tokens = set(tokens(sel_norm))

    # Heuristic color mapping
    colour_tokens = {"den", "đen", "black"}
    if (colour_tokens & q_tokens) and ("0266" in sel_tokens):
        for c_tok in colour_tokens & q_tokens:
            variant_norm = _normalise_for_synonyms(c_tok)
            canonical_norm = "0266"
            if synonyms_map.get(variant_norm) != canonical_norm:
                suggestions.append((variant_norm, canonical_norm))

    # LLM-based suggestions
    suggestions.extend(_generate_synonyms_via_llm(query, selected_name))

    # Deduplicate
    seen: Set[Tuple[str, str]] = set()
    unique = []
    for v, c in suggestions:
        key = (v, c)
        if key not in seen and synonyms_map.get(v) != c:
            unique.append(key)
            seen.add(key)

    return unique


def _generate_synonyms_via_llm(query: str, selected_name: str) -> List[Tuple[str, str]]:
    """LLM-based synonym generation."""
    pairs: List[Tuple[str, str]] = []

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return pairs

        from google import genai

        prompt = (
            f"You are a Vietnamese natural-language researcher specialising in motorbike parts. "
            f"A customer searched for: '{query}'. The product they selected was: '{selected_name}'. "
            f"Identify terms in the query that refer to the same concept as terms in the product name, "
            f"but using different words, abbreviations or romanised forms. For each such relationship, "
            f"output an object with keys 'variant' and 'canonical', where 'variant' is the term from the "
            f"query and 'canonical' is the corresponding term from the product name. If there are no new pairs, "
            f"return an empty list. Output only a JSON list of these objects with no extra commentary."
        )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = getattr(response, "text", None)

        if text:
            text_stripped = text.strip()
            start_idx = text_stripped.find('[')
            end_idx = text_stripped.rfind(']')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text_stripped[start_idx:end_idx + 1]
                parsed = json.loads(json_str)

                for obj in parsed:
                    variant_raw = str(obj.get("variant", "")).strip()
                    canonical_raw = str(obj.get("canonical", "")).strip()
                    if variant_raw and canonical_raw:
                        pairs.append((
                            _normalise_for_synonyms(variant_raw),
                            _normalise_for_synonyms(canonical_raw)
                        ))
    except Exception:
        pass

    return pairs


def append_synonym_to_file(variant_norm: str, canonical_norm: str,
                          synonyms_file: str = DEFAULT_SYNONYMS_FILE) -> None:
    """Append synonym to file."""
    row = [
        canonical_norm, "synonym_suggested", variant_norm, "", "", "", "",
        "strip_accents,lower,collapse_space", "vi", "0.5", "auto", "generated"
    ]

    try:
        file_exists = os.path.isfile(synonyms_file)
        with open(synonyms_file, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "canonical_term", "type", "variants_pipe", "brand", "model_or_series",
                    "category", "token_level_map", "normalize_rule", "language",
                    "confidence_0_1", "source", "notes"
                ])
            writer.writerow(row)
    except Exception:
        pass

    synonyms_map[variant_norm] = canonical_norm


# -----------------------------------------------------------------------------
# Edit-distance similarity + approximate substring (Bitap-style) helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=20000)
def _levenshtein_norm(a: str, b: str) -> float:
    """Normalized similarity in [0,1]. Uses rapidfuzz if present, else a banded DP fallback."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            return _RFLev.normalized_similarity(a, b)
        except Exception:
            pass
    dist = _lev_banded(a, b, max_k=None)
    m = max(len(a), len(b))
    return 1.0 - (dist / m if m else 0.0)


def _lev_banded(a: str, b: str, max_k: Optional[int] = None) -> int:
    """Levenshtein distance with optional Ukkonen band limit."""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    if max_k is None:
        max_k = la + lb
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        lo = max(1, i - max_k)
        hi = min(lb, i + max_k)
        cur = prev[:]
        cur[0] = i
        for j in range(1, lo):
            cur[j] = max_k + 1
        for j in range(lo, hi + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        for j in range(hi + 1, lb + 1):
            cur[j] = max_k + 1
        prev = cur
        if min(prev) > max_k:
            return max_k + 1
    return prev[lb]


def _best_substring_lev(pattern: str, text: str, k: int) -> int:
    """
    Approximate Bitap: best Levenshtein distance between pattern and any substring of text.
    Uses windows of len(pattern) ± 20%. For short queries only.
    """
    lp, lt = len(pattern), len(text)
    if lp == 0: return 0
    if lt == 0: return lp
    pad = max(1, int(round(lp * 0.2)))
    best = k + 1
    for L in range(max(1, lp - pad), min(lt, lp + pad) + 1):
        for i in range(0, lt - L + 1):
            sub = text[i:i+L]
            d = _lev_banded(pattern, sub, max_k=best)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------

def load_csv(csv_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load CSV with auto-detection."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


def ensure_column(headers: List[str], col_name: str) -> str:
    """Case-insensitive column lookup."""
    def norm(s: str) -> str:
        return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", s or "").lower()).strip()

    hmap = {norm(h): h for h in headers}
    key = hmap.get(norm(col_name))
    if not key:
        raise ValueError(f"Column '{col_name}' not found. Available: {headers}")
    return key


# -----------------------------------------------------------------------------
# VNSegNgramSearch (Optimized)
# -----------------------------------------------------------------------------

class VNSegNgramSearch:
    """Optimized Vietnamese n-gram search engine."""

    def __init__(self, df, name_col: str = "Tên hàng", code_col: str = "Mã hàng"):
        import pandas as pd

        if name_col not in df.columns or code_col not in df.columns:
            raise ValueError(f"DataFrame must include '{name_col}' and '{code_col}' columns.")

        self.df = df.reset_index(drop=True).copy()
        self.name_col = name_col
        self.code_col = code_col
        self.N = len(self.df)
        self._tokenizer_name = self._detect_tokenizer()

        # Precompute normalized text for all documents
        self._precompute_normalized_data()
        self._build_index()

    @staticmethod
    def _detect_tokenizer() -> str:
        """Detect available tokenizer."""
        try:
            import underthesea
            return "underthesea"
        except Exception:
            try:
                import pyvi
                return "pyvi"
            except Exception:
                return "fallback"

    def _precompute_normalized_data(self) -> None:
        """Precompute normalized text for all documents."""
        self.normalized_names = []
        self.normalized_codes = []

        for idx, row in self.df.iterrows():
            name = str(row.get(self.name_col, "") or "")
            code = str(row.get(self.code_col, "") or "")
            self.normalized_names.append(synonyms_normalize(name))
            self.normalized_codes.append(synonyms_normalize(code))

    def _segment_fallback(self, s: str) -> List[str]:
        """Optimized fallback segmentation."""
        size_pattern = r"\d+(?:/\d+)+(?:-\d+)?|\d+(?:-\d+)+"
        sizes = re.findall(size_pattern, s)

        if not sizes:
            return s.split()

        placeholder = "__SIZE__"
        temp = re.sub(size_pattern, placeholder, s)
        tokens_ = temp.split()

        final: List[str] = []
        si = 0
        for t in tokens_:
            if t == placeholder:
                final.append(sizes[si])
                si += 1
            else:
                parts = re.split(r"(?<=\D)-|-(?=\D)", t)
                final.extend(p for p in parts if p)

        return final

    def _word_tokenize(self, s: str) -> List[str]:
        """Optimized word tokenization."""
        s = synonyms_normalize(s)

        if self._tokenizer_name == "underthesea":
            try:
                from underthesea import word_tokenize
                return word_tokenize(s, format="text").split()
            except Exception:
                return self._segment_fallback(s)
        elif self._tokenizer_name == "pyvi":
            try:
                from pyvi import ViTokenizer
                return ViTokenizer.tokenize(s).replace("_", " ").split()
            except Exception:
                return self._segment_fallback(s)
        else:
            return self._segment_fallback(s)

    @staticmethod
    def _token_ngrams(tokens: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
        """Generate token n-grams."""
        token_len = len(tokens)
        if token_len < n:
            ngrams: Set[Tuple[str, ...]] = set()
            for t in tokens:
                ngrams.add((t,))
            for i in range(token_len - 1):
                ngrams.add((tokens[i], tokens[i + 1]))
            return ngrams or ({tuple(tokens)} if tokens else set())

        return {tuple(tokens[i:i + n]) for i in range(token_len - n + 1)}

    def _char_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams."""
        res: Set[str] = set()

        for fold in (False, True):
            t = WHITESPACE_RE.sub("", normalize_text(text) if fold else text.lower())
            if not t:
                continue

            if len(t) < n:
                res.add(t)
            else:
                res.update(t[i:i + n] for i in range(len(t) - n + 1))

        return res

    def _expand_choices(self, tokens: List[str]) -> List[List[str]]:
        """Expand tokens with synonyms."""
        static_syn: Dict[str, List[str]] = {
            "ju": ["jupiter", "ya", "yamaha"],
            "jupi": ["jupiter", "ya", "yamaha"],
            "jupiter": ["ju", "ya", "yamaha"],
            "ya": ["yamaha", "jupiter"],
            "xịn": ["or", "chinh", "chinhhang", "chinh_hang", "genuine", "xin"],
            "xin": ["or", "chinh", "chinhhang", "genuine", "xịn"],
            "or": ["chinhhang", "genuine", "xịn", "xin"],
            "den": ["0266"],
            "đen": ["0266"],
            "black": ["0266"],
            "0266": ["den", "black"],
        }

        choices: List[List[str]] = []

        for t in tokens:
            expansions = [t]

            # Static synonyms
            if t in static_syn:
                expansions.extend(static_syn[t])

            # External synonyms
            if synonyms_map:
                canonical = synonyms_map.get(t)
                if canonical:
                    for var, can in synonyms_map.items():
                        if can == canonical and var != t:
                            expansions.append(var)

            # Deduplicate
            seen = set()
            uniq = []
            for x in expansions:
                if x not in seen:
                    uniq.append(x)
                    seen.add(x)

            choices.append(uniq)

        return choices

    def _query_token_trigrams(self, tokens: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
        """Generate query n-grams with synonym expansion."""
        from itertools import product

        choices = self._expand_choices(tokens)

        if len(choices) < n:
            ngrams: Set[Tuple[str, ...]] = set()
            for i, choice_list in enumerate(choices):
                for a in choice_list:
                    ngrams.add((a,))
            for i in range(len(choices) - 1):
                for a in choices[i]:
                    for b in choices[i + 1]:
                        ngrams.add((a, b))
            return ngrams

        ngrams: Set[Tuple[str, ...]] = set()
        for i in range(len(choices) - n + 1):
            window = choices[i:i + n]
            for combo in product(*window):
                ngrams.add(tuple(combo))

        return ngrams

    def _build_index(self) -> None:
        """Build inverted indices."""
        self.token_postings: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
        self.char_postings: Dict[str, Set[int]] = defaultdict(set)
        self.doc_token_ngrams: List[Set[Tuple[str, ...]]] = []
        self.doc_char_ngrams: List[Set[str]] = []

        for idx in range(len(self.df)):
            name_norm = self.normalized_names[idx]
            code_norm = self.normalized_codes[idx]

            toks = self._word_tokenize(name_norm)
            t3 = self._token_ngrams(toks, n=3)
            c3 = self._char_ngrams(name_norm, n=3) | self._char_ngrams(code_norm, n=3)

            self.doc_token_ngrams.append(t3)
            self.doc_char_ngrams.append(c3)

            for ng in t3:
                self.token_postings[ng].add(idx)
            for ng in c3:
                self.char_postings[ng].add(idx)

        # Precompute IDF
        self.token_idf: Dict[Tuple[str, ...], float] = {
            ng: (math.log((self.N + 1) / (len(docs) + 1)) + 1.0)
            for ng, docs in self.token_postings.items()
        }
        self.char_idf: Dict[str, float] = {
            ng: (math.log((self.N + 1) / (len(docs) + 1)) + 1.0)
            for ng, docs in self.char_postings.items()
        }

    @staticmethod
    def _tfidf_vector(ngrams_set: Set, idf_dict: Dict) -> Dict:
        """Create TF-IDF vector."""
        return {ng: idf_dict.get(ng, 1.0) for ng in ngrams_set}

    @staticmethod
    def _cosine(d1: Dict, d2: Dict) -> float:
        """Cosine similarity."""
        if not d1 or not d2:
            return 0.0

        dot = sum(d1.get(k, 0.0) * d2.get(k, 0.0) for k in set(d1) | set(d2))
        n1 = math.sqrt(sum(v * v for v in d1.values()))
        n2 = math.sqrt(sum(v * v for v in d2.values()))

        return 0.0 if n1 == 0 or n2 == 0 else dot / (n1 * n2)

    def SEARCH_NGRAMS(self, query: str, topk: int = 1000) -> List[int]:
        """Fast candidate retrieval."""
        q_norm = synonyms_normalize(query)
        qtoks = self._word_tokenize(q_norm)
        q_t3 = self._query_token_trigrams(qtoks, n=3)
        q_c3 = self._char_ngrams(q_norm, n=3)

        counts: Counter = Counter()

        for ng in q_t3:
            for d in self.token_postings.get(ng, ()):
                counts[d] += 3

        for ng in q_c3:
            for d in self.char_postings.get(ng, ()):
                counts[d] += 1

        return [doc_id for doc_id, _ in counts.most_common(topk)]

    def SCORE_NGRAMS(self, query: str, doc_id: int, w_token: float = 0.6,
                     w_char: float = 0.4) -> float:
        """Score a document."""
        q_norm = synonyms_normalize(query)
        qtoks = self._word_tokenize(q_norm)
        q_t3 = self._query_token_trigrams(qtoks, n=3)
        q_c3 = self._char_ngrams(q_norm, n=3)

        q_vec_token = self._tfidf_vector(q_t3, self.token_idf)
        q_vec_char = self._tfidf_vector(q_c3, self.char_idf)
        d_vec_token = self._tfidf_vector(self.doc_token_ngrams[doc_id], self.token_idf)
        d_vec_char = self._tfidf_vector(self.doc_char_ngrams[doc_id], self.char_idf)

        s_token = self._cosine(q_vec_token, d_vec_token)
        s_char = self._cosine(q_vec_char, d_vec_char)

        name = str(self.df.at[doc_id, self.name_col])
        implies_genuine = any(t in qtoks for t in ("xin", "xịn", "or", "genuine", "chinhhang", "chinh", "chinh_hang"))

        if implies_genuine and GENUINE_RE.search(name):
            s_token += 0.05

        return w_token * s_token + w_char * s_char

    def search(self, query: str, topk: int = 15):
        """Search and return DataFrame."""
        import pandas as pd

        cands = self.SEARCH_NGRAMS(query, topk=topk * 50)
        scored = [(doc_id, self.SCORE_NGRAMS(query, doc_id)) for doc_id in cands]
        scored.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame([
            {
                "Score": round(score, 3),
                self.code_col: self.df.at[doc_id, self.code_col],
                self.name_col: self.df.at[doc_id, self.name_col],
                "Idx": doc_id,
            }
            for doc_id, score in scored[:topk]
        ])


# -----------------------------------------------------------------------------
# HybridSearch (Optimized & FIXED)
# -----------------------------------------------------------------------------

class HybridSearch:
    """Optimized hybrid search with domain-aware scoring - FIXED VERSION."""

    def __init__(self, df, name_col: str = "Tên hàng", code_col: str = "Mã hàng"):
        import pandas as pd

        self.df = df.reset_index(drop=True).copy()
        self.name_col = name_col
        self.code_col = code_col
        self.ng = VNSegNgramSearch(self.df, name_col=name_col, code_col=code_col)

        # Precompute document features
        self._precompute_doc_features()

    def _precompute_doc_features(self) -> None:
        """Precompute document features for faster scoring."""
        self.doc_tokens = []
        self.doc_norm_names = []
        self.doc_norm_codes = []
        self.doc_names = []  # FIX: Added missing attribute
        self.doc_codes = []  # FIX: Added missing attribute
        self.doc_has_genuine = []
        self.doc_has_0266 = []
        self.doc_sides = []

        for idx, row in self.df.iterrows():
            name = str(row.get(self.name_col, "") or "")
            code = str(row.get(self.code_col, "") or "")
            norm_name = normalize_text(name)
            norm_code = normalize_text(code)

            self.doc_names.append(name)  # FIX: Store original name
            self.doc_codes.append(code)  # FIX: Store original code
            self.doc_tokens.append(tokens(name))
            self.doc_norm_names.append(norm_name)
            self.doc_norm_codes.append(norm_code)
            self.doc_has_genuine.append(genuine_hint(name))
            self.doc_has_0266.append(("0266" in name) or ("0266" in code))
            self.doc_sides.append(detect_side(name, code))

    def score(self, query: str, doc_id: int) -> float:
        """
        Hybrid score with synonym-aware edit distance + short-query Bitap bonus.
        FIXED: Now uses self.doc_names and self.doc_codes correctly.
        """
        # Use precomputed cached values
        name = self.doc_names[doc_id]  # FIX: Now exists
        code = self.doc_codes[doc_id]  # FIX: Now exists
        nn = self.doc_norm_names[doc_id]
        cn = self.doc_norm_codes[doc_id]

        # Base scores
        s_ng = self.ng.SCORE_NGRAMS(query, doc_id)
        s_jc = jaccard(tokens(query), self.doc_tokens[doc_id])

        # Synonym-canonicalized fuzzy using true edit distance
        q_syn = synonyms_normalize(query)
        n_syn = synonyms_normalize(name)
        c_syn = synonyms_normalize(code)
        s_fz = max(_levenshtein_norm(q_syn, n_syn), _levenshtein_norm(q_syn, c_syn))

        # Dynamic weights by query length
        qlen = len(q_syn)
        if qlen <= 15:
            w_ng, w_fz, w_jc = 0.5, 0.35, 0.15
        else:
            w_ng, w_fz, w_jc = 0.5, 0.3, 0.2

        score = w_ng * s_ng + w_fz * s_fz + w_jc * s_jc

        # Domain-specific boosts
        q_norm = normalize_text(query)
        q_tokens = tokens(query)

        # Size matching
        q_size = extract_size(q_norm)
        d_size = extract_size(nn)
        if q_size and d_size and q_size == d_size:
            score += 0.08

        # Position matching (front/rear)
        q_has_front = has_front(q_norm)
        q_has_rear = has_rear(q_norm)
        d_has_front = has_front(nn)
        d_has_rear = has_rear(nn)

        if q_has_front and d_has_front:
            score += 0.04
        if q_has_rear and d_has_rear:
            score += 0.04
        if (q_has_front and d_has_rear) or (q_has_rear and d_has_front):
            score -= 0.06

        # Side matching
        q_side = detect_side(query, "")
        d_side = self.doc_sides[doc_id]
        if q_side and d_side:
            if q_side == d_side:
                score += 0.05
            else:
                score -= 0.08

        # Brand matching
        q_brands = brand_hits(q_norm)
        d_brands = brand_hits(nn)
        if q_brands > 0 and d_brands > 0:
            score += 0.03

        # Genuine hint
        implies_genuine = any(t in q_tokens for t in ("xin", "xịn", "or", "genuine", "chinhhang"))
        if implies_genuine and self.doc_has_genuine[doc_id]:
            score += 0.04

        # Color matching (0266 = black)
        q_has_black = any(t in q_tokens for t in ("den", "đen", "black", "0266"))
        if q_has_black and self.doc_has_0266[doc_id]:
            score += 0.03

        # Field-aware code similarity bonus
        code_sim = _levenshtein_norm(q_syn, c_syn)
        if code_sim >= 0.90:
            score += 0.06
        elif code_sim >= 0.80:
            score += 0.035

        # Bitap-style substring bonus for short queries
        if qlen <= 20:
            k = max(1, round(qlen * 0.12))  # edit tolerance
            d_name = _best_substring_lev(q_syn, n_syn, k)
            if d_name <= k:
                score += 0.03 * (1 - d_name / max(qlen, 1))
            d_code = _best_substring_lev(q_syn, c_syn, k)
            if d_code <= k:
                score += 0.05 * (1 - d_code / max(qlen, 1))

        return score

    def search(self, query: str, topk: int = 15):
        """Search and return ranked results."""
        import pandas as pd

        cands = self.ng.SEARCH_NGRAMS(query, topk=2000)
        scored = [(doc, self.score(query, doc)) for doc in cands]
        scored.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame([
            {
                "Score": round(s, 3),
                self.code_col: self.df.at[doc, self.code_col],
                self.name_col: self.df.at[doc, self.name_col],
                "Idx": doc,
            }
            for doc, s in scored[:topk]
        ])


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to product CSV (UTF-8)")
    ap.add_argument("--query", required=False, help="Single query to test")
    ap.add_argument("--input-file", required=False, help="Path to a text file with one query per line.")
    ap.add_argument("--output-file", default="order.txt", help="Path to save the output tab-separated file.")
    ap.add_argument("--topk", type=int, default=10, help="TopK results")
    ap.add_argument("--code-col", default="Mã hàng", help="Column name for product code")
    ap.add_argument("--name-col", default="Tên hàng", help="Column name for product name")
    ap.add_argument("--synonyms", default=DEFAULT_SYNONYMS_FILE, help="Path to synonyms CSV file")
    args = ap.parse_args()

    import pandas as pd

    try:
        load_synonyms(args.synonyms)
    except Exception:
        pass

    df = pd.read_csv(args.csv, encoding="utf-8")
    hs = HybridSearch(df, name_col=args.name_col, code_col=args.code_col)

    if args.query:
        res = hs.search(args.query, topk=args.topk)
        print(res.to_string(index=False))
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]

        results = []
        for query in queries:
            print(f"\nSearching for: {query}")
            res = hs.search(query, topk=1)
            if not res.empty:
                best_match = res.iloc[0]
                print(f"  Best match: {best_match[args.code_col]} - {best_match[args.name_col]} (Score: {best_match['Score']})")
                try:
                    quantity = int(input("  Enter quantity (0 to skip): "))
                    if quantity > 0:
                        best_match['Số lượng'] = quantity
                        results.append(best_match)
                except ValueError:
                    print("  Invalid quantity. Skipping.")
            else:
                print("  No match found.")

        if results:
            out_df = pd.DataFrame(results)
            out_df.to_csv(args.output_file, sep="\t", index=False, columns=[args.code_col, args.name_col, 'Số lượng'])
            print(f"\nOrder saved to {args.output_file}")
        else:
            print("\nNo items were ordered.")
    else:
        print("Provide --query for a single test, or --input-file for batch processing.")


if __name__ == "__main__":
    _cli()
