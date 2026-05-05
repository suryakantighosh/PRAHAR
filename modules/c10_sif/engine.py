"""
prahar/modules/c10_sif/engine.py
C-10 Stylometric Identity Fingerprint (SIF) — orchestrator.

Pipeline for one case:
  1. Load all raw_data records for the case
  2. Extract user-written text (bio, commit messages, social posts, news quotes)
  3. Pull additional writing corpus from GitHub if a username seed exists
  4. Compute 256-dim SFV via features.extract_sfv()
  5. Persist to stylometric_vector table
  6. Patch every IdentityFragment.sfv_ref for the case
  7. Return similarity dict suitable for AMCE L3 input

Comparison mode:
  compute_sif_similarity(case_id_a, case_id_b) → float [0,1]
"""

import asyncio
from typing import Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from prahar.core.db import AsyncSessionLocal
from prahar.models.raw_data import RawData
from prahar.models.identity import IdentityFragment
from prahar.modules.c01_ingestion.seed import make_case_id
from prahar.modules.c10_sif.features import extract_sfv, sfv_similarity
from prahar.modules.c10_sif.github_enricher import fetch_github_writing_corpus


# ── ORM model (inline — mirrors init.sql) ─────────────────────────────────────

from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase


class _Base(DeclarativeBase):
    pass


class StylometricVector(_Base):
    __tablename__ = "stylometric_vector"

    id         = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id    = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    entity_id  = Column(PG_UUID(as_uuid=True))      # optional — linked entity
    sfv        = Column(Vector(256))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ── Text extraction helpers ────────────────────────────────────────────────────

_TEXT_SOURCES = {
    # GitHub profile / repo data (from C-01 fetch_github_user)
    "github": lambda c: _github_text(c),
    # News / gazette snippets (from C-04)
    "google_news":  lambda c: c.get("snippet", "") + " " + c.get("title", ""),
    "duckduckgo":   lambda c: c.get("snippet", ""),
    "gazette":      lambda c: c.get("title", ""),
    # Generic OSINT — any 'content' field that is a string
}


def _github_text(content: dict) -> str:
    """Extract user-written strings from a stored GitHub payload."""
    parts = []
    profile = content.get("profile") or {}
    for field in ("bio", "company", "blog", "name"):
        v = profile.get(field) or ""
        if v:
            parts.append(v)
    for repo in content.get("repos") or []:
        desc = repo.get("description") or ""
        if desc:
            parts.append(desc)
    return "\n".join(parts)


def _extract_text_from_raw(record: RawData) -> str:
    """Pull stylometry-relevant text from one raw_data row."""
    content = record.content or {}
    source  = record.source_name or ""

    # Try known-source extractors first
    for key, extractor in _TEXT_SOURCES.items():
        if key in source.lower():
            return extractor(content)

    # Fallback: walk any string values recursively
    parts: list[str] = []

    def _walk(obj):
        if isinstance(obj, str) and len(obj) > 30:
            parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(content)
    return " ".join(parts)


def _find_github_username(records: list[RawData]) -> Optional[str]:
    """
    Scan raw_data to find a GitHub username seed so we can enrich
    directly from the API.
    """
    for rec in records:
        if "github" in (rec.source_name or "").lower():
            content = rec.content or {}
            username = (
                content.get("username")
                or (content.get("profile") or {}).get("login")
            )
            if username:
                return str(username)
    return None


# ── Main engine functions ──────────────────────────────────────────────────────

async def compute_sfv_for_case(
    case_id: UUID,
    enrich_github: bool = True,
) -> dict:
    """
    Compute and persist a Stylometric Feature Vector for the given case.

    Returns:
        {
          "case_id":    str,
          "sfv_id":     str | None,
          "text_chars": int,
          "sfv_norm":   float,       # L2 norm before normalisation (quality indicator)
          "success":    bool,
        }
    """
    logger.info(f"[C-10] Computing SFV for case={case_id}")

    # ── 1. Load raw data ───────────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        stmt = select(RawData).where(RawData.case_id == case_id)
        rows = (await db.execute(stmt)).scalars().all()

    if not rows:
        logger.warning(f"[C-10] No raw_data for case={case_id}")
        return {"case_id": str(case_id), "sfv_id": None,
                "text_chars": 0, "sfv_norm": 0.0, "success": False}

    # ── 2. Build text corpus ───────────────────────────────────────────────────
    corpus_parts: list[str] = [_extract_text_from_raw(r) for r in rows]

    # ── 3. GitHub enrichment ───────────────────────────────────────────────────
    if enrich_github:
        username = _find_github_username(rows)
        if username:
            logger.info(f"[C-10] Enriching from GitHub: {username}")
            gh_corpus = await fetch_github_writing_corpus(username)
            if gh_corpus:
                corpus_parts.append(gh_corpus)

    full_text = "\n\n".join(p for p in corpus_parts if p.strip())
    logger.info(f"[C-10] Corpus built: {len(full_text):,} chars")

    # ── 4. Compute SFV ────────────────────────────────────────────────────────
    import numpy as np
    loop = asyncio.get_event_loop()
    sfv  = await loop.run_in_executor(None, extract_sfv, full_text)

    if sfv is None:
        logger.warning(f"[C-10] Insufficient text for SFV, case={case_id}")
        return {"case_id": str(case_id), "sfv_id": None,
                "text_chars": len(full_text), "sfv_norm": 0.0, "success": False}

    sfv_norm = float(np.linalg.norm(sfv))

    # ── 5. Persist ────────────────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        record = StylometricVector(
            case_id=case_id,
            sfv=sfv.tolist(),
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        sfv_id = record.id
        logger.success(
            f"[C-10] SFV stored id={sfv_id} "
            f"norm={sfv_norm:.4f} chars={len(full_text):,}"
        )

    # ── 6. Patch IdentityFragment.sfv_ref ─────────────────────────────────────
    async with AsyncSessionLocal() as db:
        stmt = select(IdentityFragment).where(
            IdentityFragment.case_id == case_id,
            IdentityFragment.sfv_ref.is_(None),
        )
        fragments = (await db.execute(stmt)).scalars().all()
        for frag in fragments:
            frag.sfv_ref = sfv_id
        await db.commit()
        logger.info(f"[C-10] Patched sfv_ref on {len(fragments)} fragments")

    return {
        "case_id":    str(case_id),
        "sfv_id":     str(sfv_id),
        "text_chars": len(full_text),
        "sfv_norm":   sfv_norm,
        "success":    True,
    }


async def compute_sif_similarity(
    case_id_a: UUID,
    case_id_b: UUID,
) -> float:
    """
    Load stored SFVs for two cases and return cosine similarity [0, 1].
    Returns 0.5 (neutral / unknown) if either case has no SFV yet.
    """
    import numpy as np

    async def _load_sfv(cid: UUID):
        async with AsyncSessionLocal() as db:
            stmt = (
                select(StylometricVector)
                .where(StylometricVector.case_id == cid)
                .order_by(StylometricVector.created_at.desc())
                .limit(1)
            )
            row = (await db.execute(stmt)).scalars().first()
            if row and row.sfv:
                return np.array(row.sfv, dtype=np.float32)
            return None

    sfv_a, sfv_b = await asyncio.gather(_load_sfv(case_id_a), _load_sfv(case_id_b))

    if sfv_a is None or sfv_b is None:
        logger.warning(
            f"[C-10] Missing SFV for similarity: "
            f"a={'ok' if sfv_a is not None else 'missing'} "
            f"b={'ok' if sfv_b is not None else 'missing'}"
        )
        return 0.5   # neutral — same as ARF KL-divergence default

    sim = sfv_similarity(sfv_a, sfv_b)
    logger.info(f"[C-10] SIF similarity {case_id_a} ↔ {case_id_b} = {sim:.4f}")
    return sim


async def get_sfv_for_case(case_id: UUID):
    """
    Return the most recent SFV numpy array for a case, or None.
    Used by C-12 optimizer and C-02 CPIF enrichment.
    """
    import numpy as np
    async with AsyncSessionLocal() as db:
        stmt = (
            select(StylometricVector)
            .where(StylometricVector.case_id == case_id)
            .order_by(StylometricVector.created_at.desc())
            .limit(1)
        )
        row = (await db.execute(stmt)).scalars().first()
        if row and row.sfv:
            return np.array(row.sfv, dtype=np.float32)
    return None
