"""
prahar/modules/c04_records/engine.py
C-04 Public Records Crawler — orchestrator.
Runs all 4 sources concurrently, writes graph edges to Neo4j.
"""
import asyncio
from typing import Optional
from uuid import UUID
from loguru import logger
import aiohttp

from prahar.modules.c04_records.connectors import (
    search_mca21, search_ecourts,
    search_gazette, search_google_news, dork_search,
)
from prahar.modules.c01_ingestion.seed import make_case_id
from prahar.core.db import AsyncSessionLocal
from prahar.models.public_record import PublicRecord, NewsRecord


async def crawl_public_records(
    subject_name: str,
    case_id: Optional[UUID] = None,
) -> dict:
    """
    Full public records pipeline for a subject name.
    Runs MCA21, eCourts, Gazette, Google News concurrently.
    Writes legal graph edges to Neo4j (via C-06 when built).
    """
    if case_id is None:
        case_id = make_case_id()

    logger.info(f"[C-04] Crawling public records for '{subject_name}' case={case_id}")

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as session:

        # Playwright tasks run in thread pool (blocking)
        loop = asyncio.get_event_loop()

        mca_task    = loop.run_in_executor(None, asyncio.run,
                          search_mca21(subject_name))
        ecourt_task = loop.run_in_executor(None, asyncio.run,
                          search_ecourts(subject_name))

        # Async tasks run directly
        gazette_task = search_gazette(subject_name, session)
        news_task    = search_google_news(subject_name, session)
        dork_task    = dork_search(
            f'"{subject_name}" site:mca.gov.in OR site:ecourts.gov.in',
            session
        )

        mca_results, ecourt_results, gazette_results, news_results, dork_results = (
            await asyncio.gather(
                asyncio.wrap_future(mca_task),
                asyncio.wrap_future(ecourt_task),
                gazette_task,
                news_task,
                dork_task,
                return_exceptions=True,
            )
        )

        # Normalise exceptions to empty lists
        def safe(r):
            return r if isinstance(r, list) else []

        mca_results     = safe(mca_results)
        ecourt_results  = safe(ecourt_results)
        gazette_results = safe(gazette_results)
        news_results    = safe(news_results)
        dork_results    = safe(dork_results)

        # Persist to DB
        async with AsyncSessionLocal() as db:
            saved_records = 0
            saved_news    = 0

            for rec in mca_results + ecourt_results + gazette_results + dork_results:
                pr = PublicRecord(
                    case_id=case_id,
                    record_type=rec.get("source", "unknown").upper(),
                    source=rec.get("source", ""),
                    subject=subject_name,
                    content=rec,
                    source_url=rec.get("url", ""),
                )
                db.add(pr)
                saved_records += 1

            for article in news_results:
                nr = NewsRecord(
                    case_id=case_id,
                    title=article.get("title"),
                    snippet=article.get("snippet"),
                    source_url=article.get("url"),
                    publisher=article.get("publisher"),
                    published=article.get("published"),
                )
                db.add(nr)
                saved_news += 1

            await db.commit()

    summary = {
        "case_id":         str(case_id),
        "subject":         subject_name,
        "mca21":           len(mca_results),
        "ecourts":         len(ecourt_results),
        "gazette":         len(gazette_results),
        "news":            len(news_results),
        "dork":            len(dork_results),
        "saved_records":   saved_records,
        "saved_news":      saved_news,
    }

    logger.success(
        f"[C-04] '{subject_name}' — "
        f"mca={len(mca_results)} courts={len(ecourt_results)} "
        f"news={len(news_results)} gazette={len(gazette_results)}"
    )
    return summary
