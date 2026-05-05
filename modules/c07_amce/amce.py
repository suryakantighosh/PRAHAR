"""
prahar/modules/c07_amce/amce.py
Adaptive Multi-Layer Confidence Engine (AMCE) — Novel Patent Method.

4-layer Bayesian scoring:
  L1 — Raw signal scoring       (how many sources confirm this identity?)
  L2 — Structural corroboration (does the graph topology support it?)
  L3 — Behavioral alignment     (do temporal/stylometric patterns match?)
  L4 — Conflict penalty         (how many CONTRADICTS edges touch this node?)

Final score = weighted sum of L1–L4, weights updated by C-12 optimizer.
"""
import math
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger


# Default layer weights (sum to 1.0)
DEFAULT_W_L1 = 0.35
DEFAULT_W_L2 = 0.30
DEFAULT_W_L3 = 0.20
DEFAULT_W_L4 = 0.15   # penalty layer — subtracted

RISK_THRESHOLDS = {
    "HIGH":   0.80,
    "MEDIUM": 0.55,
    "LOW":    0.35,
}


@dataclass
class AMCEInput:
    """All signals needed to compute AMCE score for one identity."""
    identity_id:      str

    # L1 inputs
    source_count:     int   = 0     # number of distinct sources confirming
    platform_count:   int   = 0     # number of platforms found on
    breach_count:     int   = 0     # number of data breaches found

    # L2 inputs
    graph_degree:     int   = 0     # number of edges in Neo4j
    corroboration_score: float = 0.0  # fraction of claims with 2+ sources

    # L3 inputs
    cpif_score:       float = 0.0   # from C-02
    sif_similarity:   float = 0.0   # from C-10 (stylometric)
    tbs_kl_score:     float = 0.5   # from C-11 (temporal, 0.5 = unknown)

    # L4 inputs
    conflict_count:   int   = 0     # number of CONTRADICTS edges
    conflict_weight:  float = 0.0   # total weight of conflicts

    # Metadata
    risk_flags:       List[str] = field(default_factory=list)


@dataclass
class AMCEResult:
    identity_id:  str
    score_l1:     float
    score_l2:     float
    score_l3:     float
    score_l4:     float   # penalty (positive = bad)
    final_score:  float
    risk_level:   str
    risk_flags:   List[str]
    breakdown:    Dict[str, Any]


# ── Layer scoring functions ───────────────────────────────────

def score_l1_raw_signals(inp: AMCEInput) -> float:
    """
    L1: Raw signal scoring.
    More sources + more platforms = higher base confidence.
    Uses log-saturation to prevent gaming via source stuffing.
    """
    # Log-saturation: diminishing returns after ~10 sources
    source_score   = min(1.0, math.log1p(inp.source_count)   / math.log1p(20))
    platform_score = min(1.0, math.log1p(inp.platform_count) / math.log1p(15))
    breach_bonus   = min(0.15, inp.breach_count * 0.03)  # breaches add certainty

    return round(
        0.5 * source_score + 0.35 * platform_score + 0.15 * breach_bonus,
        4
    )


def score_l2_structural(inp: AMCEInput) -> float:
    """
    L2: Structural corroboration.
    Well-connected graph nodes with cross-source support score higher.
    """
    degree_score = min(1.0, math.log1p(inp.graph_degree) / math.log1p(50))
    corr_score   = min(1.0, inp.corroboration_score)

    return round(0.4 * degree_score + 0.6 * corr_score, 4)


def score_l3_behavioral(inp: AMCEInput) -> float:
    """
    L3: Behavioral alignment.
    CPIF fusion score + stylometric fingerprint + temporal rhythm alignment.
    """
    cpif_contrib = inp.cpif_score                     # already in [0,1]
    sif_contrib  = inp.sif_similarity                 # already in [0,1]
    tbs_contrib  = inp.tbs_kl_score                   # already in [0,1]

    return round(
        0.40 * cpif_contrib +
        0.30 * sif_contrib  +
        0.30 * tbs_contrib,
        4
    )


def score_l4_conflict_penalty(inp: AMCEInput) -> float:
    """
    L4: Conflict penalty.
    Each CONTRADICTS edge reduces final score.
    Higher conflict weight = larger penalty.
    Returns penalty value in [0, 1] — subtracted from final score.
    """
    if inp.conflict_count == 0:
        return 0.0

    count_penalty  = min(0.5, inp.conflict_count * 0.08)
    weight_penalty = min(0.5, inp.conflict_weight * 0.15)
    return round(min(0.8, count_penalty + weight_penalty), 4)


def compute_risk_flags(inp: AMCEInput, final_score: float) -> List[str]:
    """Generate human-readable risk flags based on signal values."""
    flags = list(inp.risk_flags)  # start with any pre-set flags

    if inp.breach_count >= 3:
        flags.append("MULTIPLE_BREACHES")
    if inp.breach_count >= 1:
        flags.append("DATA_BREACH_EXPOSURE")
    if inp.conflict_count >= 3:
        flags.append("HIGH_DATA_CONFLICT")
    if inp.platform_count >= 20:
        flags.append("EXTENSIVE_DIGITAL_FOOTPRINT")
    if inp.cpif_score >= 0.90:
        flags.append("HIGH_CONFIDENCE_IDENTITY_FUSION")
    if final_score >= RISK_THRESHOLDS["HIGH"]:
        flags.append("HIGH_CONFIDENCE_SUBJECT")
    if inp.sif_similarity >= 0.85:
        flags.append("STRONG_STYLOMETRIC_MATCH")
    if inp.tbs_kl_score >= 0.85:
        flags.append("STRONG_TEMPORAL_RHYTHM_MATCH")

    return list(dict.fromkeys(flags))  # dedup while preserving order


# ── Main AMCE scorer ─────────────────────────────────────────

def compute_amce(
    inp: AMCEInput,
    w_l1: float = DEFAULT_W_L1,
    w_l2: float = DEFAULT_W_L2,
    w_l3: float = DEFAULT_W_L3,
    w_l4: float = DEFAULT_W_L4,
) -> AMCEResult:
    """
    Compute full 4-layer AMCE score for one identity.
    Weights are updated by C-12 AMCE Optimizer via gradient descent.
    """
    l1 = score_l1_raw_signals(inp)
    l2 = score_l2_structural(inp)
    l3 = score_l3_behavioral(inp)
    l4 = score_l4_conflict_penalty(inp)

    # Weighted sum minus conflict penalty
    raw_score = w_l1 * l1 + w_l2 * l2 + w_l3 * l3 - w_l4 * l4
    final = round(max(0.0, min(1.0, raw_score)), 4)

    # Risk level classification
    if final >= RISK_THRESHOLDS["HIGH"]:
        risk_level = "HIGH"
    elif final >= RISK_THRESHOLDS["MEDIUM"]:
        risk_level = "MEDIUM"
    elif final >= RISK_THRESHOLDS["LOW"]:
        risk_level = "LOW"
    else:
        risk_level = "VERY_LOW"

    flags = compute_risk_flags(inp, final)

    logger.info(
        f"[AMCE] identity={inp.identity_id} "
        f"L1={l1:.3f} L2={l2:.3f} L3={l3:.3f} L4={l4:.3f} "
        f"final={final:.3f} risk={risk_level}"
    )

    return AMCEResult(
        identity_id=inp.identity_id,
        score_l1=l1,
        score_l2=l2,
        score_l3=l3,
        score_l4=l4,
        final_score=final,
        risk_level=risk_level,
        risk_flags=flags,
        breakdown={
            "weights": {"w_l1": w_l1, "w_l2": w_l2,
                        "w_l3": w_l3, "w_l4": w_l4},
            "contributions": {
                "l1_weighted": round(w_l1 * l1, 4),
                "l2_weighted": round(w_l2 * l2, 4),
                "l3_weighted": round(w_l3 * l3, 4),
                "l4_penalty":  round(w_l4 * l4, 4),
            },
        },
    )


def batch_score(
    inputs: List[AMCEInput],
    w_l1: float = DEFAULT_W_L1,
    w_l2: float = DEFAULT_W_L2,
    w_l3: float = DEFAULT_W_L3,
    w_l4: float = DEFAULT_W_L4,
) -> List[AMCEResult]:
    """Score multiple identities, sorted by final_score descending."""
    results = [compute_amce(inp, w_l1, w_l2, w_l3, w_l4) for inp in inputs]
    return sorted(results, key=lambda r: r.final_score, reverse=True)
