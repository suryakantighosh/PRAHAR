"""
prahar/modules/c05_nlp/engine.py
C-05 NLP Pipeline — orchestrator.
Processes all raw_data text for a case through NER,
deduplicates entities, resolves co-references,
and persists to entity master table.
"""
import asyncio
from typing import Optional, List
from uuid import UUID
from loguru import logger

from prahar.modules.c05_nlp.ner import (
    extract_entities, merge_entity_lists,
    basic_coref_resolve, ExtractedEntity,
)
from prahar.modules.c01_ingestion.seed import make_case_id
from prahar.core.db import AsyncSessionLocal
from prahar.models.nlp import Entity, EntityAlias


def _is_base64(s):
    return isinstance(s, str) and len(s) > 100 and not any(c in s for c in [' ', '.', ','])

def _extract_text_from_payload(payload: dict) -> str:
    """
    Pull readable text from a raw_data content payload.
    Handles different source formats (news, whois, crt.sh etc.).
    """
    texts = []

    # Recursive text extractor
    def _is_junk(s):
        if len(s)>150 and " " not in s[:80]: return True
        if s.startswith("http") and len(s)>60: return True
        if s.startswith("data:image"): return True
        return False
    def _walk(obj):
        if isinstance(obj, str) and len(obj)>20 and not _is_junk(obj):
            texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)
    return " ".join(texts)[:50000]   # cap at 50k chars per record


async def process_case_text(
    case_id: UUID,
) -> dict:
    """
    Load all raw_data for a case, run NER on each record,
    merge into entity master list, persist to DB.
    """
    from sqlalchemy import select
    from prahar.models.raw_data import RawData

    logger.info(f"[C-05] Processing NLP for case={case_id}")

    # Load all raw records for this case
    async with AsyncSessionLocal() as db:
        stmt = select(RawData).where(RawData.case_id == case_id)
        rows = (await db.execute(stmt)).scalars().all()

    if not rows:
        logger.warning(f"[C-05] No raw data found for case={case_id}")
        return {"case_id": str(case_id), "entities": 0, "records_processed": 0}

    # Run NER in thread pool (CPU-heavy)
    loop = asyncio.get_event_loop()

    def _run_ner_batch():
        all_lists = []
        for row in rows:
            text = _extract_text_from_payload(row.content or {})
            if not text.strip():
                continue
            entities = extract_entities(text, source_id=str(row.id))
            all_lists.append(entities)
        return all_lists

    all_entity_lists = await loop.run_in_executor(None, _run_ner_batch)

    # Merge + co-ref resolve
    merged = merge_entity_lists(all_entity_lists)
    resolved = basic_coref_resolve(merged)

    # Persist to DB
    async with AsyncSessionLocal() as db:
        for ent in resolved:
            db_entity = Entity(
                case_id=case_id,
                text=ent.text,
                label=ent.label,
                canonical_form=ent.canonical_form,
                meta={
                    "count":   ent.count,
                    "sources": ent.sources[:10],   # cap for storage
                },
            )
            db.add(db_entity)
            await db.flush()   # get entity_id

            for alias in ent.aliases:
                db.add(EntityAlias(
                    entity_id=db_entity.entity_id,
                    alias_text=alias,
                ))

        await db.commit()

    logger.success(
        f"[C-05] case={case_id} records={len(rows)} "
        f"entities={len(resolved)}"
    )
    return {
        "case_id":           str(case_id),
        "records_processed": len(rows),
        "entities":          len(resolved),
        "top_persons":       [
            e.canonical_form for e in resolved
            if e.label == "PERSON"
        ][:5],
        "top_orgs":          [
            e.canonical_form for e in resolved
            if e.label == "ORG"
        ][:5],
    }
