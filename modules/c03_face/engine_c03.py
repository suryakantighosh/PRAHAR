"""
prahar/modules/c03_face/engine.py
C-03 Face Engine — orchestrator.
Embeds faces, stores in pgvector, matches against case gallery.
"""
import asyncio
from typing import Optional, List
from uuid import UUID
import aiohttp
from loguru import logger

from prahar.modules.c03_face.ensemble import embed_face, ensemble_match, FaceEmbeddingResult
from prahar.modules.c01_ingestion.seed import make_case_id
from prahar.core.db import AsyncSessionLocal
from prahar.models.face import FaceEmbedding, FaceMatch


async def _download_image(url: str, session: aiohttp.ClientSession) -> Optional[bytes]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                return await r.read()
    except Exception as e:
        logger.warning(f"[C-03] Download failed {url}: {e}")
    return None


async def process_face_url(
    url: str,
    case_id: UUID,
    session: aiohttp.ClientSession,
) -> Optional[FaceEmbedding]:
    """Download image, embed, persist to DB. Returns ORM object or None."""
    img_bytes = await _download_image(url, session)
    if not img_bytes:
        return None

    # Run CPU-heavy embedding in thread pool (keeps event loop free)
    loop = asyncio.get_event_loop()
    emb_result = await loop.run_in_executor(None, embed_face, img_bytes)

    if emb_result.rejected:
        logger.warning(f"[C-03] Rejected image: {url}")
        return None

    async with AsyncSessionLocal() as db:
        face = FaceEmbedding(
            case_id=case_id,
            source_url=url,
            embedding_arcface=(
                emb_result.arcface.tolist() if emb_result.arcface is not None else None
            ),
            embedding_insight=(
                emb_result.insightface.tolist() if emb_result.insightface is not None else None
            ),
            embedding_openface=(
                emb_result.dlib.tolist() if emb_result.dlib is not None else None
            ),
            exif_meta=emb_result.exif_meta,
            blur_score=emb_result.blur_score,
        )
        db.add(face)
        await db.commit()
        await db.refresh(face)
        logger.success(f"[C-03] Stored embedding id={face.id} url={url[:60]}")
        return face


async def match_faces_in_case(case_id: UUID) -> List[dict]:
    """
    Load all face embeddings for a case and run pairwise ensemble matching.
    Stores confirmed matches in face_match table.
    Returns list of match dicts.
    """
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        stmt = select(FaceEmbedding).where(FaceEmbedding.case_id == case_id)
        rows = (await db.execute(stmt)).scalars().all()

    if len(rows) < 2:
        logger.info(f"[C-03] Less than 2 faces for case {case_id} — no matching needed")
        return []

    import numpy as np

    def row_to_emb(row: FaceEmbedding) -> FaceEmbeddingResult:
        r = FaceEmbeddingResult()
        if row.embedding_arcface:
            r.arcface = np.array(row.embedding_arcface, dtype=np.float32)
        if row.embedding_insight:
            r.insightface = np.array(row.embedding_insight, dtype=np.float32)
        if row.embedding_openface:
            r.dlib = np.array(row.embedding_openface, dtype=np.float32)
        r.blur_score = row.blur_score or 0.0
        return r

    matches_found = []
    async with AsyncSessionLocal() as db:
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                emb_a = row_to_emb(rows[i])
                emb_b = row_to_emb(rows[j])
                result = ensemble_match(emb_a, emb_b)

                if result["match"]:
                    fm = FaceMatch(
                        case_id=case_id,
                        source_a=rows[i].id,
                        source_b=rows[j].id,
                        similarity_score=result["similarity_score"],
                        consensus_count=result["consensus_count"],
                        confirmed=False,
                    )
                    db.add(fm)
                    matches_found.append({
                        "face_a": str(rows[i].id),
                        "face_b": str(rows[j].id),
                        **result,
                    })
        await db.commit()

    logger.success(
        f"[C-03] case={case_id} faces={len(rows)} matches={len(matches_found)}"
    )
    return matches_found
