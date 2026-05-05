"""
prahar/modules/c10_sif/features.py
Stylometric Identity Fingerprint (SIF) — 256-dim feature vector.

Vector layout (256 total):
  [0:32]   Lexical richness & surface statistics
  [32:132] Function word frequencies (100 words)
  [132:148] Punctuation / special-char densities (16)
  [148:164] Universal POS-tag frequencies (16) — zero if spaCy unavailable
  [164:216] Character trigram fingerprint (52)
  [216:256] Sentence & word-length histograms (40)

All values are L2-normalised before return so cosine similarity
is just a dot product.
"""

import re
import math
from collections import Counter
from typing import Optional

import numpy as np
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────

# 100 high-frequency English function words (exactly 100 — verified)
FUNCTION_WORDS: list[str] = [
    "the","be","to","of","and","a","in","that","have","it",     # 10
    "for","not","on","with","he","as","you","do","at","this",   # 10
    "but","his","by","from","they","we","say","her","she","or", # 10
    "an","will","my","one","all","would","there","their","what","so", # 10
    "up","out","if","about","who","get","which","go","me","when",     # 10
    "make","can","like","no","just","him","know","into","your","good", # 10
    "some","could","them","see","other","than","then","now","look","only", # 10
    "come","its","over","think","also","back","after","use","two","how",   # 10
    "our","work","first","well","way","even","new","want","because","any",  # 10
    "give","day","most","us","am","are","were","been","time","year",        # 10
]

# 16 Universal POS tags (spaCy)
UNIVERSAL_POS: list[str] = [
    "NOUN","VERB","ADJ","ADV","PRON","DET","ADP",
    "NUM","CONJ","PART","INTJ","PUNCT","SYM","X","SPACE","PROPN",
]

# 16 punctuation characters to track
PUNCT_CHARS: list[str] = [
    ".","," ,"!","?",";",":","'",'"',"-","_","(",")","/","@","#","*",
]

# Top 52 English character trigrams (reference fingerprint)
CHAR_TRIGRAMS: list[str] = [
    "the","and","ing","ion","tio","ent","ati","for","her","ter",
    "hat","tha","ere","con","res","ver","all","ons","nce","men",
    "ith","ted","ers","pro","thi","wit","are","ess","not","ive",
    "was","ect","rea","com","eve","per","int","est","sta","tin",
    "ist","ble","his","had","our","ome","hin","ove","ort","whi",
    "ear","tic",
]

# ── Lazy spaCy loader ──────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_trf")
        logger.info("[C-10/SIF] spaCy en_core_web_trf loaded")
        return _nlp
    except Exception as e:
        logger.warning(f"[C-10/SIF] spaCy unavailable — POS block will be zeros: {e}")
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Split text into sentences (regex-based, no model needed)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _words(text: str) -> list[str]:
    """Lowercase alphabetic tokens only."""
    return re.findall(r"[a-z]+", text.lower())


# ── Block A: Lexical richness (32 features) ───────────────────────────────────

def _block_lexical(text: str, words: list[str], sents: list[str]) -> np.ndarray:
    vec = np.zeros(32, dtype=np.float32)
    if not words:
        return vec

    n_w = len(words)
    n_s = max(len(sents), 1)
    lengths = np.array([len(w) for w in words], dtype=np.float32)
    sent_lens = np.array([len(s.split()) for s in sents], dtype=np.float32)

    vec[0]  = float(lengths.mean())
    vec[1]  = float(lengths.std())
    vec[2]  = len(set(words)) / n_w                           # type-token ratio
    cnt = Counter(words)
    vec[3]  = sum(1 for c in cnt.values() if c == 1) / max(len(cnt), 1)  # hapax ratio
    vec[4]  = float(sent_lens.mean())
    vec[5]  = float(sent_lens.std())
    vec[6]  = sum(len(s) for s in sents) / n_s               # avg chars/sentence
    vec[7]  = n_s / (n_w / 100.0)                            # sentences per 100 words
    vec[8]  = sum(1 for w in words if len(w) <= 3) / n_w    # short word ratio
    vec[9]  = sum(1 for w in words if 4 <= len(w) <= 7) / n_w
    vec[10] = sum(1 for w in words if len(w) >= 8) / n_w    # long word ratio
    # uppercase + digit + punctuation density over all chars
    all_chars = text
    nc = max(len(all_chars), 1)
    vec[11] = sum(1 for c in all_chars if c.isupper()) / nc
    vec[12] = sum(1 for c in all_chars if c.isdigit()) / nc
    vec[13] = sum(1 for c in all_chars if not c.isalnum() and not c.isspace()) / nc
    vec[14] = all_chars.count("!") / nc
    vec[15] = all_chars.count("?") / nc
    # Yule's K (vocabulary richness measure)
    freq = Counter(cnt.values())
    M1   = n_w
    M2   = sum(v * (v ** 2) for v in cnt.values())
    vec[16] = (10000 * (M2 - M1) / (M1 ** 2)) if M1 > 0 else 0.0
    # Brunet's W
    vec[17] = (n_w ** (len(cnt) ** -0.172)) if n_w > 0 else 0.0
    # Honore's R
    hapax = sum(1 for c in cnt.values() if c == 1)
    vec[18] = (100 * math.log(n_w) / (1 - hapax / max(len(cnt), 1))) if n_w > 1 else 0.0
    # avg word rank (Zipf proxy) — skipped, use comma density
    vec[19] = all_chars.count(",") / nc
    vec[20] = all_chars.count(";") / nc
    vec[21] = all_chars.count(":") / nc
    vec[22] = all_chars.count("-") / nc
    vec[23] = all_chars.count("(") / nc
    # 24-31: sentence length variance bins (capped)
    for i, cap in enumerate([5, 10, 15, 20, 30, 50, 75, 1000]):
        prev = [5,10,15,20,30,50,75,0][i]
        vec[24 + i] = sum(1 for l in sent_lens if prev < l <= cap) / n_s
    return vec


# ── Block B: Function word frequencies (100 features) ─────────────────────────

def _block_function_words(words: list[str]) -> np.ndarray:
    if not words:
        return np.zeros(100, dtype=np.float32)
    cnt = Counter(words)
    n = len(words)
    return np.array([cnt.get(fw, 0) / n for fw in FUNCTION_WORDS], dtype=np.float32)


# ── Block C: Punctuation densities (16 features) ──────────────────────────────

def _block_punctuation(text: str) -> np.ndarray:
    nc = max(len(text), 1)
    return np.array(
        [text.count(p) / nc for p in PUNCT_CHARS],
        dtype=np.float32,
    )


# ── Block D: POS-tag frequencies (16 features) ────────────────────────────────

def _block_pos(text: str) -> np.ndarray:
    nlp = _get_nlp()
    if nlp is None:
        return np.zeros(16, dtype=np.float32)
    try:
        doc = nlp(text[:10000])     # cap for speed
        total = max(len(doc), 1)
        cnt = Counter(tok.pos_ for tok in doc)
        return np.array(
            [cnt.get(tag, 0) / total for tag in UNIVERSAL_POS],
            dtype=np.float32,
        )
    except Exception as e:
        logger.warning(f"[C-10/SIF] POS extraction failed: {e}")
        return np.zeros(16, dtype=np.float32)


# ── Block E: Character trigram fingerprint (52 features) ─────────────────────

def _block_char_trigrams(text: str) -> np.ndarray:
    text_lower = text.lower()
    n = max(len(text_lower) - 2, 1)
    cnt = Counter(text_lower[i:i+3] for i in range(len(text_lower) - 2))
    total = max(sum(cnt.values()), 1)
    return np.array(
        [cnt.get(tg, 0) / total for tg in CHAR_TRIGRAMS],
        dtype=np.float32,
    )


# ── Block F: Word & sentence length histograms (40 features) ─────────────────

def _block_histograms(words: list[str], sents: list[str]) -> np.ndarray:
    vec = np.zeros(40, dtype=np.float32)
    # Word length histogram: buckets 1–9, 10+  (10 features)
    n_w = max(len(words), 1)
    for i, length in enumerate(range(1, 10)):
        vec[i] = sum(1 for w in words if len(w) == length) / n_w
    vec[9] = sum(1 for w in words if len(w) >= 10) / n_w
    # Sentence length histogram: 1-5, 6-10, 11-15, 16-20, 21-30, 31-40, 41-60, 61-80, 81-100, 100+
    n_s = max(len(sents), 1)
    bins = [(1,5),(6,10),(11,15),(16,20),(21,30),(31,40),(41,60),(61,80),(81,100),(101,10000)]
    for i, (lo, hi) in enumerate(bins):
        vec[10 + i] = sum(1 for s in sents if lo <= len(s.split()) <= hi) / n_s
    # Char-per-word histogram: 1-3, 4-5, 6-7, 8-9, 10-12, 12+  (6 features)
    for i, (lo, hi) in enumerate([(1,3),(4,5),(6,7),(8,9),(10,12),(13,999)]):
        vec[20 + i] = sum(1 for w in words if lo <= len(w) <= hi) / n_w
    # Avg chars per sentence quartile features (4)
    if sents:
        clens = sorted(len(s) for s in sents)
        q = len(clens) // 4 or 1
        vec[26] = sum(clens[:q]) / (q * 100)
        vec[27] = sum(clens[q:2*q]) / (q * 100)
        vec[28] = sum(clens[2*q:3*q]) / (q * 100)
        vec[29] = sum(clens[3*q:]) / (max(len(clens) - 3*q, 1) * 100)
    # Padding: bigram POS proxy via word-start-char patterns (10 features)
    if words:
        n_w2 = max(len(words) - 1, 1)
        for i in range(10):
            vec[30 + i] = sum(
                1 for j in range(len(words) - 1)
                if (ord(words[j][0]) + ord(words[j+1][0])) % 10 == i
            ) / n_w2
    return vec


# ── Main entry point ───────────────────────────────────────────────────────────

MIN_TEXT_LEN = 50    # minimum chars to produce a meaningful vector

def extract_sfv(text: str) -> Optional[np.ndarray]:
    """
    Compute a 256-dim L2-normalised Stylometric Feature Vector.
    Returns None if text is too short to be meaningful.
    """
    if not text or len(text) < MIN_TEXT_LEN:
        return None

    text  = text[:200_000]           # hard cap — avoid memory issues
    words = _words(text)
    sents = _sentences(text)

    if len(words) < 10:
        return None

    parts = [
        _block_lexical(text, words, sents),     # 32
        _block_function_words(words),            # 100
        _block_punctuation(text),                # 16
        _block_pos(text),                        # 16
        _block_char_trigrams(text),              # 52
        _block_histograms(words, sents),         # 40
    ]
    # Equalise block contributions before concatenation — prevents the
    # high-dim function-word block from drowning out smaller but highly
    # discriminative blocks (e.g. punctuation, lexical richness).
    normed = []
    for blk in parts:
        n = np.linalg.norm(blk)
        normed.append(blk / n if n > 0 else blk)
    vec = np.concatenate(normed).astype(np.float32)
    assert vec.shape == (256,), f"SFV shape error: {vec.shape}"

    # Replace NaN / Inf
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def sfv_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised SFVs.
    Returns float in [-1, 1], clamped to [0, 1] for practical use.
    """
    if a is None or b is None:
        return 0.0
    sim = float(np.dot(a, b))
    return max(0.0, min(1.0, sim))
