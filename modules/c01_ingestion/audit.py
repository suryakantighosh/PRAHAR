"""
prahar/modules/c01_ingestion/audit.py
Centralised audit logger — every scraped record gets
SHA-256 hash + provenance written to PostgreSQL.
"""
import hashlib
import json
from datetime import datetime
from uuid import UUID
from typing import Union

from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from prahar.models.raw_data import RawData


def sha256_content(content: Union[dict, str]) -> str:
    """Stable SHA-256 of any content (dict serialised to sorted JSON)."""
    if isinstance(content, dict):
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False)
    else:
        raw = str(content)
    return hashlib.sha256(raw.encode()).hexdigest()


async def store_record(
    db: AsyncSession,
    *,
    case_id: UUID,
    seed_hash: str,
    source_url: str,
    source_name: str,
    content: dict,
    robots_allowed: bool = True,
) -> RawData:
    """
    Persist one scraped record with full provenance.
    Returns the saved ORM object (includes auto-generated id).
    """
    record = RawData(
        case_id=case_id,
        seed_hash=seed_hash,
        source_url=source_url,
        source_name=source_name,
        content=content,
        content_hash=sha256_content(content),
        fetched_at=datetime.utcnow(),
        robots_allowed=robots_allowed,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    logger.info(
        f"[AUDIT] case={case_id} src={source_name} "
        f"hash={record.content_hash[:12]}… url={source_url[:60]}"
    )
    return record
