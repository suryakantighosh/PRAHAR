"""
prahar/modules/c11_tbs/engine.py
C-11 Temporal Behavioral Scoring (TBS) — async orchestrator.

Pipeline for one case:
  1. Load all raw_data timestamps for the case from PostgreSQL
  2. Compute 64-dim TBP via features.extract_tbp()
  3. Persist to temporal_profile table
  4. Return tbs_kl_score suitable for AMCE L3 input

Comparison mode:
  compute_tbs_similarity(case_id_a, case_id_b)
    → float [0, 1] via tbp_kl_score_from_timestamps (JSD on raw ts)

The engine stores the normalised TBP vector for fast bulk comparison and
also caches the raw timestamp list (as a sorted JSON array of ISO strings)
so the JSD path can be used when comparing two cases later.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from sqlalchemy import Column, DateTime, Text, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase

from prahar.core.db import AsyncSessionLocal
from prahar.models.raw_data import RawData
from prahar.modules.c11_tbs.features import (
    MIN_EVENTS,
    extract_tbp,
    tbp_kl_score_from_timestamps,
)

try:
    from pgvector.sqlalchemy import Vector as PgVector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    logger.warning("[C-11/TBS] pgvector not available — TBP stored as JSON text")


# ── ORM model ─────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class TemporalProfile(_Base):
    """
    Persisted temporal activity profile for one case.

    tbp_vector  — 64-dim L2-normalised float32 array (stored as pgvector
                  when available, else as JSON text fallback).
    ts_cache    — JSON array of ISO-format UTC timestamp strings used for
                  JSD recomputation without re-querying raw_data.
    event_count — number of raw timestamps used; quality indicator.
    """
    __tablename__ = "temporal_profile"

    id          = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id     = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    event_count = Column(Text, nullable=False, default="0")   # stored as str
    ts_cache    = Column(Text, nullable=True)                  # JSON list of ISO strings
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    # tbp_vector: use pgvector(64) when available, fall back to JSON text
    if _HAS_PGVECTOR:
        tbp_vector = Column(PgVector(64))
    else:
        tbp_vector = Column(Text)   # JSON fallback


# ── Timestamp extraction from raw_data ────────────────────────────────────────

def _parse_ts(value: object) -> Optional[datetime]:
    """
    Attempt to parse a timestamp from a raw_data content value.
    Handles ISO-8601 strings and unix epoch integers.
    Returns a naive UTC datetime or None.
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.utcfromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _walk_for_timestamps(obj: object, depth: int = 0) -> list[datetime]:
    """
    Recursively walk a raw_data content dict looking for anything that
    looks like a timestamp.  Stops at depth 6 to avoid pathological nesting.
    """
    if depth > 6:
        return []

    results: list[datetime] = []

    if isinstance(obj, dict):
        for key, val in obj.items():
            # High-value keys get parsed directly
            if any(k in key.lower() for k in ("created", "updated", "posted",
                                               "published", "timestamp", "date",
                                               "time", "at")):
                ts = _parse_ts(val)
                if ts:
                    results.append(ts)
            else:
                results.extend(_walk_for_timestamps(val, depth + 1))

    elif isinstance(obj, list):
        for item in obj:
            results.extend(_walk_for_timestamps(item, depth + 1))

    elif isinstance(obj, (str, int, float)):
        ts = _parse_ts(obj)
        if ts:
            results.append(ts)

    return results


def _extract_timestamps_from_record(record: RawData) -> list[datetime]:
    """
    Extract all timestamp candidates from one raw_data row.
    Always includes the record's own created_at as a reliable fallback.
    """
    candidates: list[datetime] = []

    # The record's own DB timestamp is always valid
    if record.fetched_at:
        candidates.append(record.fetched_at.replace(tzinfo=None))

    # Walk the content dict
    content = record.content or {}
    candidates.extend(_walk_for_timestamps(content))

    # Deduplicate and filter implausible dates
    seen = set()
    MIN_TS = datetime(2005, 1, 1)
    MAX_TS = datetime(2035, 1, 1)
    out: list[datetime] = []
    for ts in candidates:
        key = ts.replace(microsecond=0)   # 1-second granularity dedup
        if key not in seen and MIN_TS <= ts <= MAX_TS:
            seen.add(key)
            out.append(ts)

    return out


# ── Persistence helpers ────────────────────────────────────────────────────────

def _serialise_tbp(vec: np.ndarray) -> object:
    """Convert numpy float32 array to storage-appropriate type."""
    if _HAS_PGVECTOR:
        return vec.tolist()
    return json.dumps(vec.tolist())


def _deserialise_tbp(stored: object) -> Optional[np.ndarray]:
    """Restore stored TBP back to numpy float32 array."""
    if stored is None:
        return None
    if isinstance(stored, str):
        try:
            return np.array(json.loads(stored), dtype=np.float32)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(stored, list):
        return np.array(stored, dtype=np.float32)
    return None


# ── Main engine functions ──────────────────────────────────────────────────────

async def compute_tbp_for_case(case_id: UUID) -> dict:
    """
    Compute and persist a Temporal Behavioral Profile for the given case.

    Returns
    -------
    {
      "case_id":     str,
      "profile_id":  str | None,
      "event_count": int,
      "success":     bool,
    }
    """
    logger.info(f"[C-11/TBS] Computing TBP for case={case_id}")

    # ── 1. Load raw_data rows ─────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        stmt = select(RawData).where(RawData.case_id == case_id)
        rows = (await db.execute(stmt)).scalars().all()

    if not rows:
        logger.warning(f"[C-11/TBS] No raw_data found for case={case_id}")
        return {"case_id": str(case_id), "profile_id": None,
                "event_count": 0, "success": False}

    # ── 2. Extract timestamps ─────────────────────────────────────────────────
    all_ts: list[datetime] = []
    for row in rows:
        all_ts.extend(_extract_timestamps_from_record(row))

    logger.info(f"[C-11/TBS] Extracted {len(all_ts)} timestamps from "
                f"{len(rows)} raw_data rows")

    if len(all_ts) < 1:
        logger.warning(
            f"[C-11/TBS] Insufficient timestamps ({len(all_ts)}) "
            f"for case={case_id}, minimum={MIN_EVENTS}"
        )
        return {"case_id": str(case_id), "profile_id": None,
                "event_count": len(all_ts), "success": False}

    # ── 3. Compute TBP vector (CPU-bound — run in thread) ────────────────────
    loop = asyncio.get_event_loop()
    tbp_vec = await loop.run_in_executor(None, extract_tbp, all_ts)

    if tbp_vec is None:
        logger.warning(
            f"[C-11/TBS] extract_tbp returned None for case={case_id}"
        )
        return {"case_id": str(case_id), "profile_id": None,
                "event_count": len(all_ts), "success": False}

    # ── 4. Persist ────────────────────────────────────────────────────────────
    ts_cache_json = json.dumps([ts.isoformat() for ts in sorted(all_ts)])

    async with AsyncSessionLocal() as db:
        record = TemporalProfile(
            case_id=case_id,
            event_count=str(len(all_ts)),
            ts_cache=ts_cache_json,
            tbp_vector=_serialise_tbp(tbp_vec),
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        profile_id = record.id

    logger.success(
        f"[C-11/TBS] TBP stored id={profile_id} "
        f"events={len(all_ts)} for case={case_id}"
    )

    return {
        "case_id":     str(case_id),
        "profile_id":  str(profile_id),
        "event_count": len(all_ts),
        "success":     True,
    }


async def _load_profile(case_id: UUID) -> Optional[TemporalProfile]:
    """Load the most recent TemporalProfile row for a case."""
    async with AsyncSessionLocal() as db:
        stmt = (
            select(TemporalProfile)
            .where(TemporalProfile.case_id == case_id)
            .order_by(TemporalProfile.created_at.desc())
            .limit(1)
        )
        return (await db.execute(stmt)).scalars().first()


async def compute_tbs_similarity(
    case_id_a: UUID,
    case_id_b: UUID,
) -> float:
    """
    Load stored temporal profiles for two cases and return the TBS
    similarity score for AMCE L3 input.

    Uses tbp_kl_score_from_timestamps (JSD on raw distributions) when
    the cached timestamp lists are available — this is the semantically
    correct path.  Falls back to cosine similarity on stored TBP vectors
    if the cache is missing.  Returns 0.5 (neutral) if either case has
    no profile yet.

    Returns
    -------
    float in [0, 1]  — 1.0 = identical rhythms, 0.5 = unknown
    """
    profile_a, profile_b = await asyncio.gather(
        _load_profile(case_id_a),
        _load_profile(case_id_b),
    )

    if profile_a is None or profile_b is None:
        missing = []
        if profile_a is None: missing.append(str(case_id_a))
        if profile_b is None: missing.append(str(case_id_b))
        logger.warning(
            f"[C-11/TBS] Missing profiles for: {', '.join(missing)} — returning 0.5"
        )
        return 0.5

    # ── Preferred path: JSD on raw timestamps ────────────────────────────────
    try:
        if profile_a.ts_cache and profile_b.ts_cache:
            ts_a = [
                datetime.fromisoformat(s)
                for s in json.loads(profile_a.ts_cache)
            ]
            ts_b = [
                datetime.fromisoformat(s)
                for s in json.loads(profile_b.ts_cache)
            ]
            score = tbp_kl_score_from_timestamps(ts_a, ts_b)
            logger.info(
                f"[C-11/TBS] JSD score {case_id_a} ↔ {case_id_b} = {score:.4f}"
            )
            return score
    except Exception as exc:
        logger.warning(f"[C-11/TBS] JSD path failed ({exc}), falling back to cosine")

    # ── Fallback: cosine on stored TBP vectors ────────────────────────────────
    vec_a = _deserialise_tbp(profile_a.tbp_vector)
    vec_b = _deserialise_tbp(profile_b.tbp_vector)

    if vec_a is None or vec_b is None:
        return 0.5

    score = float(max(0.0, min(1.0, np.dot(vec_a, vec_b))))
    logger.info(
        f"[C-11/TBS] cosine score {case_id_a} ↔ {case_id_b} = {score:.4f}"
    )
    return score


async def get_tbp_for_case(case_id: UUID) -> Optional[np.ndarray]:
    """
    Return the most recent TBP numpy array for a case, or None.
    Used by C-12 optimizer.
    """
    profile = await _load_profile(case_id)
    if profile is None:
        return None
    return _deserialise_tbp(profile.tbp_vector)
