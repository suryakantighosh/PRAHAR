"""
prahar/modules/c02_identity/uvht.py
Username Variant Hash Tree (UVHT)
Builds a trie of semantic username variants for CPIF matching.
Variants: dots, underscores, numbers, case, prefix/suffix patterns.
"""
import re
from typing import Dict, List, Tuple
import numpy as np


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def generate_variants(username: str) -> List[str]:
    """
    Generate likely username variants for a given base username.
    Returns deduped list including the original.
    """
    base = username.lower().strip()
    variants = {base}

    # Separator substitutions
    for sep_from, sep_to in [("_", "."), (".", "_"), ("_", ""), (".", "")]:
        variants.add(base.replace(sep_from, sep_to))

    # Strip leading/trailing digits
    stripped = re.sub(r"^\d+|\d+$", "", base)
    if stripped and stripped != base:
        variants.add(stripped)

    # Common suffix patterns
    for suffix in ["_", ".", "x", "official", "real", "1", "2", "01", "99"]:
        variants.add(f"{base}{suffix}")
        variants.add(f"{base.rstrip('_.')}_{suffix.strip('_.')}")

    # Common prefix patterns
    for prefix in ["the", "its", "im", "i_am", "real"]:
        variants.add(f"{prefix}{base}")
        variants.add(f"{prefix}_{base}")

    # Remove empties
    return [v for v in variants if v]


def uvht_similarity(username_a: str, username_b: str) -> float:
    """
    UVHT similarity score between two usernames.
    Returns float in [0, 1]. 1.0 = exact match after normalisation.
    Used as w_usr component in CPIF F(A,B) formula.
    """
    a = username_a.lower().strip()
    b = username_b.lower().strip()

    if a == b:
        return 1.0

    # Direct edit distance score
    max_len = max(len(a), len(b), 1)
    edit_score = 1.0 - (_edit_distance(a, b) / max_len)

    # Variant membership: does b appear in a's variant set?
    a_variants = set(generate_variants(a))
    b_variants = set(generate_variants(b))
    variant_overlap = len(a_variants & b_variants) / max(len(a_variants | b_variants), 1)

    # Combine: edit distance weighted more heavily
    return round(0.85 * edit_score + 0.15 * variant_overlap, 4)


def build_uvht(usernames: List[str]) -> Dict[str, List[str]]:
    """
    Build a Username Variant Hash Tree for a list of usernames.
    Returns dict: {canonical_username: [all_variants]}
    """
    tree = {}
    for uname in usernames:
        tree[uname.lower()] = generate_variants(uname)
    return tree
