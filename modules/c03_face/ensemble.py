"""
prahar/modules/c03_face/ensemble.py
3-model consensus engine.
Rule: accept match only if >= 2/3 models agree (cosine similarity > 0.85).
This is the ensemble voting described in PRAHAR patent claims.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from prahar.modules.c03_face.embedders import (
    embed_arcface, embed_insightface, embed_dlib
)
from prahar.modules.c03_face.preprocess import preprocess

MATCH_THRESHOLD = 0.85      # cosine similarity threshold per model
CONSENSUS_MIN   = 2         # minimum models that must agree


@dataclass
class FaceEmbeddingResult:
    """All embeddings for one face image."""
    arcface:    Optional[np.ndarray] = None   # 512-dim
    insightface: Optional[np.ndarray] = None  # 512-dim
    dlib:       Optional[np.ndarray] = None   # 128-dim
    blur_score: float = 0.0
    exif_meta:  Dict[str, Any] = field(default_factory=dict)
    models_succeeded: int = 0
    rejected: bool = False


def cosine_sim(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Cosine similarity — vectors assumed L2-normalised."""
    if v_a is None or v_b is None:
        return 0.0
    dot = float(np.dot(v_a, v_b))
    return max(-1.0, min(1.0, dot))   # clamp numerical errors


def embed_face(img_bytes: bytes) -> FaceEmbeddingResult:
    """
    Full embedding pipeline for one image:
    1. Preprocess (EXIF fix, blur check, resize)
    2. Run all 3 models
    3. Return FaceEmbeddingResult
    """
    result = FaceEmbeddingResult()

    img, meta = preprocess(img_bytes)
    result.exif_meta = meta
    result.blur_score = meta.get("blur_score", 0.0)

    if img is None:
        result.rejected = True
        logger.warning("[C-03] Image rejected at preprocessing")
        return result

    # Run models — failures return None, don't crash
    result.arcface    = embed_arcface(img)
    result.insightface = embed_insightface(img)
    result.dlib       = embed_dlib(img)

    result.models_succeeded = sum(
        1 for v in [result.arcface, result.insightface, result.dlib]
        if v is not None
    )

    logger.info(
        f"[C-03] Embedding complete — "
        f"{result.models_succeeded}/3 models succeeded, "
        f"blur={result.blur_score:.1f}"
    )
    return result


def ensemble_match(
    emb_a: FaceEmbeddingResult,
    emb_b: FaceEmbeddingResult,
) -> Dict[str, Any]:
    """
    3-model consensus vote between two face embeddings.
    Returns match result with per-model scores and consensus count.
    """
    votes: List[Tuple[str, float, bool]] = []

    # Model 1: ArcFace
    if emb_a.arcface is not None and emb_b.arcface is not None:
        sim = cosine_sim(emb_a.arcface, emb_b.arcface)
        votes.append(("arcface", sim, sim >= MATCH_THRESHOLD))

    # Model 2: InsightFace
    if emb_a.insightface is not None and emb_b.insightface is not None:
        sim = cosine_sim(emb_a.insightface, emb_b.insightface)
        votes.append(("insightface", sim, sim >= MATCH_THRESHOLD))

    # Model 3: dlib
    if emb_a.dlib is not None and emb_b.dlib is not None:
        sim = cosine_sim(emb_a.dlib, emb_b.dlib)
        votes.append(("dlib", sim, sim >= MATCH_THRESHOLD))

    if not votes:
        return {
            "match": False,
            "consensus_count": 0,
            "similarity_score": 0.0,
            "per_model": {},
            "reason": "no_models_available",
        }

    agree_count = sum(1 for _, _, agreed in votes if agreed)
    avg_sim = sum(s for _, s, _ in votes) / len(votes)
    matched = agree_count >= CONSENSUS_MIN

    logger.info(
        f"[C-03] Ensemble: {agree_count}/{len(votes)} agree, "
        f"avg_sim={avg_sim:.3f}, match={matched}"
    )

    return {
        "match": matched,
        "consensus_count": agree_count,
        "models_voted": len(votes),
        "similarity_score": round(avg_sim, 4),
        "per_model": {
            name: {"similarity": round(sim, 4), "agreed": agreed}
            for name, sim, agreed in votes
        },
    }


def find_matches_in_set(
    query: FaceEmbeddingResult,
    gallery: List[Tuple[str, FaceEmbeddingResult]],   # (id, embedding)
) -> List[Dict[str, Any]]:
    """
    Find all gallery faces that match the query via ensemble vote.
    Returns sorted list of matches (highest similarity first).
    """
    matches = []
    for face_id, gallery_emb in gallery:
        result = ensemble_match(query, gallery_emb)
        if result["match"]:
            matches.append({"id": face_id, **result})

    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches
