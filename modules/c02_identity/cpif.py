"""
prahar/modules/c02_identity/cpif.py
Cross-Platform Identity Fusion (CPIF) — Novel Patent Method.

Formula:
  F(A,B) = w_bio * cos(v_A, v_B) + w_usr * UVHT_sim(A,B) + w_tbs * ARF_KL(A,B)
  Merge into CIN if F(A,B) > threshold θ (default 0.72, adaptive)

This is the core patentable algorithm of PRAHAR v2.
"""
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from prahar.modules.c02_identity.uvht import uvht_similarity


# Default weights (updated by C-12 AMCE Optimizer via gradient descent)
DEFAULT_W_BIO = 0.40
DEFAULT_W_USR = 0.35
DEFAULT_W_TBS = 0.25
DEFAULT_THETA = 0.72    # merge threshold


@dataclass
class IdentitySignal:
    """
    All signals for one Identity Fragment Node (IFN).
    Fields set to None when not yet collected.
    """
    fragment_id:    UUID = field(default_factory=uuid4)
    platform:       str = ""
    username:       Optional[str] = None
    email:          Optional[str] = None
    phone:          Optional[str] = None
    # Biometric vector from C-03 (512-dim ArcFace)
    biometric_vec:  Optional[np.ndarray] = None
    # ARF KL-divergence from C-11 (pre-computed, lower = more similar)
    arf_kl_score:   Optional[float] = None
    uncertainty:    float = 1.0


def cosine_similarity(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(v_a, v_b) / (norm_a * norm_b))


def arf_kl_to_score(kl_divergence: Optional[float]) -> float:
    """
    Convert KL-divergence to a [0,1] similarity score.
    Low KL-divergence = similar temporal rhythms = high score.
    Uses exponential decay: score = exp(-kl / 2)
    """
    if kl_divergence is None:
        return 0.5   # unknown → neutral
    return float(np.exp(-kl_divergence / 2.0))


def cpif_score(
    signal_a: IdentitySignal,
    signal_b: IdentitySignal,
    w_bio: float = DEFAULT_W_BIO,
    w_usr: float = DEFAULT_W_USR,
    w_tbs: float = DEFAULT_W_TBS,
) -> Dict[str, Any]:
    """
    Compute CPIF fusion score F(A,B) between two Identity Fragment Nodes.

    Returns dict with:
      - score: float in [0,1]
      - components: breakdown of each signal contribution
      - signals_used: list of which signals contributed
    """
    components = {}
    signals_used = []
    effective_w_bio = w_bio
    effective_w_usr = w_usr
    effective_w_tbs = w_tbs

    # ── Biometric component ──────────────────────────────────
    bio_score = 0.0
    if (signal_a.biometric_vec is not None and
            signal_b.biometric_vec is not None):
        bio_score = cosine_similarity(signal_a.biometric_vec,
                                       signal_b.biometric_vec)
        signals_used.append("biometric")
    else:
        # Redistribute bio weight equally to other signals
        bonus = effective_w_bio / 2.0
        effective_w_usr += bonus
        effective_w_tbs += bonus
        effective_w_bio = 0.0

    components["bio"] = round(bio_score * effective_w_bio, 4)

    # ── Username component (UVHT) ────────────────────────────
    usr_score = 0.0
    if signal_a.username and signal_b.username:
        usr_score = uvht_similarity(signal_a.username, signal_b.username)
        signals_used.append("username")
    elif signal_a.email and signal_b.email:
        # Fall back to email local-part similarity
        local_a = signal_a.email.split("@")[0]
        local_b = signal_b.email.split("@")[0]
        usr_score = uvht_similarity(local_a, local_b)
        signals_used.append("email_local")
    else:
        effective_w_tbs += effective_w_usr
        effective_w_usr = 0.0

    components["usr"] = round(usr_score * effective_w_usr, 4)

    # ── Temporal behavioral component (ARF KL-divergence) ───
    tbs_score = arf_kl_to_score(signal_a.arf_kl_score)
    if signal_a.arf_kl_score is not None:
        signals_used.append("temporal")
    components["tbs"] = round(tbs_score * effective_w_tbs, 4)

    # ── Final fusion score ───────────────────────────────────
    f_score = components["bio"] + components["usr"] + components["tbs"]
    f_score = round(min(f_score, 1.0), 4)

    return {
        "score": f_score,
        "components": components,
        "signals_used": signals_used,
        "weights": {"w_bio": effective_w_bio,
                    "w_usr": effective_w_usr,
                    "w_tbs": effective_w_tbs},
    }


def should_merge(score: float, theta: float = DEFAULT_THETA) -> bool:
    """Return True if two IFNs should be merged into a CIN."""
    return score >= theta


def fuse_fragments(
    fragments: List[IdentitySignal],
    w_bio: float = DEFAULT_W_BIO,
    w_usr: float = DEFAULT_W_USR,
    w_tbs: float = DEFAULT_W_TBS,
    theta: float = DEFAULT_THETA,
) -> List[List[IdentitySignal]]:
    """
    Greedy fusion: group all fragments that mutually exceed threshold θ.
    Returns list of groups — each group becomes one CIN.
    """
    n = len(fragments)
    # Build adjacency (merge) matrix
    merge_matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            result = cpif_score(fragments[i], fragments[j], w_bio, w_usr, w_tbs)
            if should_merge(result["score"], theta):
                merge_matrix[i][j] = True
                merge_matrix[j][i] = True

    # Union-find grouping
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if merge_matrix[i][j]:
                union(i, j)

    # Collect groups
    groups: Dict[int, List[IdentitySignal]] = {}
    for i, frag in enumerate(fragments):
        root = find(i)
        groups.setdefault(root, []).append(frag)

    return list(groups.values())
