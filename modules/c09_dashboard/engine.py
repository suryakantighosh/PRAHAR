"""
prahar/modules/c09_dashboard/engine.py
C-09 Dashboard Engine — async data aggregation layer.

Pulls live data from PostgreSQL, Neo4j, and Redis to feed the
REST API and live frontend. Designed to be cheap to call
(read-only, no heavy compute).

Exposed functions:
  get_system_health()       → infra connectivity check
  get_case_list()           → paginated case summaries
  get_case_detail(case_id)  → full case report card
  get_weight_dashboard()    → AMCE/CPIF weight history + trends
  get_pipeline_stats()      → per-module record counts
  get_entity_leaderboard()  → top entities across all cases
  get_recent_activity()     → latest raw_data records
  get_graph_stats()         → Neo4j node/edge counts
  search_cases(query)       → FTS over raw_data content
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func, text, distinct

from prahar.core.db import AsyncSessionLocal
from prahar.models.raw_data import RawData
from prahar.models.identity import IdentityFragment, ConsolidatedIdentity, BreachRecord
from prahar.models.public_record import PublicRecord, NewsRecord
from prahar.models.nlp import Entity
from prahar.models.amce import ThreatScore, SignalWeights
from prahar.modules.c12_optimizer.engine import (
    load_current_amce_weights,
    load_current_cpif_weights,
    get_weight_history,
    AMCEWeightRecord,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════════════

async def get_system_health() -> dict:
    """
    Ping all infrastructure services.
    Returns a dict of {service: status} — 'ok' | 'error: <msg>'.
    """
    results = {}

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        results["postgres"] = "ok"
    except Exception as e:
        results["postgres"] = f"error: {e}"

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        import os
        import redis.asyncio as aioredis
        r = await aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        await r.ping()
        await r.aclose()
        results["redis"] = "ok"
    except Exception as e:
        results["redis"] = f"error: {e}"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    try:
        from prahar.modules.c06_graph.driver import get_driver
        driver = await get_driver()
        async with driver.session() as session:
            await session.run("RETURN 1")
        results["neo4j"] = "ok"
    except Exception as e:
        results["neo4j"] = f"error: {e}"

    # ── Ollama ────────────────────────────────────────────────────────────────
    try:
        import aiohttp, os
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as r:
                results["ollama"] = "ok" if r.status == 200 else f"http {r.status}"
    except Exception as e:
        results["ollama"] = f"error: {e}"

    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    results["overall"] = overall
    results["checked_at"] = datetime.utcnow().isoformat()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STATS  (record counts per module)
# ══════════════════════════════════════════════════════════════════════════════

async def get_pipeline_stats() -> dict:
    """
    Count rows in each module's primary table.
    Returns dict of {module: {total, today}}.
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async with AsyncSessionLocal() as db:
        async def count_table(model, ts_col):
            total = (await db.execute(select(func.count()).select_from(model))).scalar() or 0
            since = (
                await db.execute(
                    select(func.count()).select_from(model).where(ts_col >= today_start)
                )
            ).scalar() or 0
            return {"total": total, "today": since}

        raw       = await count_table(RawData,             RawData.fetched_at)
        fragments = await count_table(IdentityFragment,    IdentityFragment.created_at)
        cins      = await count_table(ConsolidatedIdentity,ConsolidatedIdentity.created_at)
        entities  = await count_table(Entity,              Entity.created_at)
        pub_recs  = await count_table(PublicRecord,        PublicRecord.created_at)
        news      = await count_table(NewsRecord,          NewsRecord.created_at)
        scores    = await count_table(ThreatScore,         ThreatScore.created_at)
        breaches  = await count_table(BreachRecord,        BreachRecord.created_at)

    return {
        "c01_raw_data":              raw,
        "c02_identity_fragments":    fragments,
        "c02_consolidated_ids":      cins,
        "c02_breach_records":        breaches,
        "c04_public_records":        pub_recs,
        "c04_news_records":          news,
        "c05_entities":              entities,
        "c07_threat_scores":         scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CASE LIST
# ══════════════════════════════════════════════════════════════════════════════

async def get_case_list(
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "last_activity",   # last_activity | risk_score | created
) -> dict:
    """
    Return paginated list of cases with summary stats.
    A 'case' is identified by its UUID in the raw_data table.
    """
    offset = (page - 1) * page_size

    async with AsyncSessionLocal() as db:
        # Distinct case_ids with their most recent activity
        case_q = (
            select(
                RawData.case_id,
                func.count(RawData.id).label("record_count"),
                func.max(RawData.fetched_at).label("last_activity"),
                func.min(RawData.fetched_at).label("first_seen"),
            )
            .group_by(RawData.case_id)
        )

        if sort_by == "last_activity":
            case_q = case_q.order_by(func.max(RawData.fetched_at).desc())
        elif sort_by == "created":
            case_q = case_q.order_by(func.min(RawData.fetched_at).desc())

        total_cases = (
            await db.execute(
                select(func.count(distinct(RawData.case_id)))
            )
        ).scalar() or 0

        rows = (await db.execute(case_q.offset(offset).limit(page_size))).all()

        cases = []
        for row in rows:
            case_id = row.case_id

            # Get threat score if available
            ts = (
                await db.execute(
                    select(ThreatScore)
                    .where(ThreatScore.case_id == case_id)
                    .order_by(ThreatScore.created_at.desc())
                    .limit(1)
                )
            ).scalars().first()

            # Get source count
            source_count = (
                await db.execute(
                    select(func.count(distinct(RawData.source_name)))
                    .where(RawData.case_id == case_id)
                )
            ).scalar() or 0

            # Get entity count
            entity_count = (
                await db.execute(
                    select(func.count()).select_from(Entity)
                    .where(Entity.case_id == case_id)
                )
            ).scalar() or 0

            cases.append({
                "case_id":      str(case_id),
                "record_count": row.record_count,
                "source_count": source_count,
                "entity_count": entity_count,
                "first_seen":   row.first_seen.isoformat() if row.first_seen else None,
                "last_activity":row.last_activity.isoformat() if row.last_activity else None,
                "risk_level":   ts.risk_level if ts else "UNSCORED",
                "final_score":  ts.final_score if ts else None,
                "risk_flags":   ts.risk_flags if ts else [],
            })

    if sort_by == "risk_score":
        cases.sort(key=lambda c: c["final_score"] or 0, reverse=True)

    return {
        "cases":       cases,
        "total":       total_cases,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1, (total_cases + page_size - 1) // page_size),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CASE DETAIL
# ══════════════════════════════════════════════════════════════════════════════

async def get_case_detail(case_id: UUID) -> dict:
    """
    Full report card for one case — all modules' outputs.
    """
    async with AsyncSessionLocal() as db:

        # ── Raw data ──────────────────────────────────────────────────────────
        raw_rows = (
            await db.execute(
                select(RawData)
                .where(RawData.case_id == case_id)
                .order_by(RawData.fetched_at.desc())
                .limit(50)
            )
        ).scalars().all()

        sources = list({r.source_name for r in raw_rows})
        seed_hashes = list({r.seed_hash for r in raw_rows})

        # ── Identity ──────────────────────────────────────────────────────────
        fragments = (
            await db.execute(
                select(IdentityFragment).where(IdentityFragment.case_id == case_id)
            )
        ).scalars().all()

        cins = (
            await db.execute(
                select(ConsolidatedIdentity).where(ConsolidatedIdentity.case_id == case_id)
            )
        ).scalars().all()

        breaches = (
            await db.execute(
                select(BreachRecord).where(BreachRecord.case_id == case_id)
            )
        ).scalars().all()

        # ── NLP entities ─────────────────────────────────────────────────────
        entities = (
            await db.execute(
                select(Entity)
                .where(Entity.case_id == case_id)
                .order_by(Entity.created_at.desc())
                .limit(100)
            )
        ).scalars().all()

        # ── Public records ────────────────────────────────────────────────────
        pub_recs = (
            await db.execute(
                select(PublicRecord).where(PublicRecord.case_id == case_id)
            )
        ).scalars().all()

        news = (
            await db.execute(
                select(NewsRecord)
                .where(NewsRecord.case_id == case_id)
                .order_by(NewsRecord.created_at.desc())
                .limit(20)
            )
        ).scalars().all()

        # ── Threat scores ─────────────────────────────────────────────────────
        scores = (
            await db.execute(
                select(ThreatScore)
                .where(ThreatScore.case_id == case_id)
                .order_by(ThreatScore.created_at.desc())
                .limit(10)
            )
        ).scalars().all()

    latest_score = scores[0] if scores else None

    # Top persons and orgs
    persons = [e for e in entities if e.label == "PERSON"]
    orgs    = [e for e in entities if e.label == "ORG"]
    gpes    = [e for e in entities if e.label in ("GPE", "LOC")]

    return {
        "case_id": str(case_id),
        "summary": {
            "seed_hashes":        seed_hashes,
            "sources_found":      sources,
            "records_total":      len(raw_rows),
            "fragments":          len(fragments),
            "consolidated_ids":   len(cins),
            "breaches":           len(breaches),
            "entities":           len(entities),
            "public_records":     len(pub_recs),
            "news_articles":      len(news),
        },
        "risk": {
            "level":      latest_score.risk_level if latest_score else "UNSCORED",
            "score":      latest_score.final_score if latest_score else None,
            "score_l1":   latest_score.score_l1 if latest_score else None,
            "score_l2":   latest_score.score_l2 if latest_score else None,
            "score_l3":   latest_score.score_l3 if latest_score else None,
            "score_l4":   latest_score.score_l4 if latest_score else None,
            "flags":      latest_score.risk_flags if latest_score else [],
            "scored_at":  latest_score.created_at.isoformat() if latest_score else None,
        },
        "identity": {
            "platforms":    sorted({f.platform for f in fragments if f.platform}),
            "usernames":    sorted({f.username for f in fragments if f.username}),
            "emails":       sorted({f.email for f in fragments if f.email}),
            "phones":       sorted({f.phone for f in fragments if f.phone}),
            "breach_names": [b.breach_name for b in breaches if b.breach_name],
        },
        "entities": {
            "top_persons": [
                {"text": e.canonical_form or e.text,
                 "count": (e.meta or {}).get("count", 1)}
                for e in sorted(persons, key=lambda x: (x.meta or {}).get("count", 1), reverse=True)[:10]
            ],
            "top_orgs": [
                {"text": e.canonical_form or e.text,
                 "count": (e.meta or {}).get("count", 1)}
                for e in sorted(orgs, key=lambda x: (x.meta or {}).get("count", 1), reverse=True)[:10]
            ],
            "locations": [
                {"text": e.canonical_form or e.text}
                for e in gpes[:10]
            ],
        },
        "news": [
            {
                "title":     n.title,
                "snippet":   n.snippet,
                "publisher": n.publisher,
                "published": n.published,
                "url":       n.source_url,
            }
            for n in news
        ],
        "public_records": [
            {
                "type":    p.record_type,
                "source":  p.source,
                "subject": p.subject,
                "url":     p.source_url,
            }
            for p in pub_recs[:20]
        ],
        "score_history": [
            {
                "final_score": s.final_score,
                "risk_level":  s.risk_level,
                "scored_at":   s.created_at.isoformat(),
            }
            for s in scores
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

async def get_weight_dashboard() -> dict:
    """
    AMCE layer weights + CPIF signal weights + trend history.
    Designed for the live monitoring panel.
    """
    amce_w, cpif_w, history = await asyncio.gather(
        load_current_amce_weights(),
        load_current_cpif_weights(),
        get_weight_history(limit=50),
    )

    # Compute trend: compare last 5 vs previous 5
    trend = {}
    if len(history) >= 10:
        recent = history[-5:]
        prior  = history[-10:-5]

        def avg(recs, key):
            vals = [r[key] for r in recs if r[key] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        for key in ("w_l1", "w_l2", "w_l3", "w_l4"):
            delta = avg(recent, key) - avg(prior, key)
            trend[key] = "up" if delta > 0.005 else "down" if delta < -0.005 else "stable"
    else:
        trend = {k: "stable" for k in ("w_l1", "w_l2", "w_l3", "w_l4")}

    # Loss trend
    losses = [r["loss"] for r in history if r["loss"] is not None]

    async with AsyncSessionLocal() as db:
        n_feedback = (
            await db.execute(
                select(func.count()).select_from(
                    __import__("prahar.modules.c12_optimizer.engine", fromlist=["FeedbackEvent"]).FeedbackEvent
                )
            )
        ).scalar() or 0

    return {
        "current_weights": {
            "amce": {
                "w_l1": round(amce_w.w_l1, 6),
                "w_l2": round(amce_w.w_l2, 6),
                "w_l3": round(amce_w.w_l3, 6),
                "w_l4": round(amce_w.w_l4, 6),
            },
            "cpif": {
                "w_bio": round(cpif_w.w_bio, 6),
                "w_usr": round(cpif_w.w_usr, 6),
                "w_tbs": round(cpif_w.w_tbs, 6),
            },
        },
        "trends":          trend,
        "history":         history,
        "loss_series":     losses,
        "n_feedback_events": n_feedback,
        "n_optimizer_runs":  len(history),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTITY LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

async def get_entity_leaderboard(
    label: str = "PERSON",
    limit: int = 20,
) -> dict:
    """
    Top entities by frequency across all cases.
    """
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(
                    Entity.canonical_form,
                    Entity.label,
                    func.count(Entity.entity_id).label("case_count"),
                    func.sum(
                        func.cast(
                            func.coalesce(Entity.meta["count"].astext, "1"),
                            __import__("sqlalchemy").Integer
                        )
                    ).label("total_mentions"),
                )
                .where(Entity.label == label)
                .group_by(Entity.canonical_form, Entity.label)
                .order_by(func.count(Entity.entity_id).desc())
                .limit(limit)
            )
        ).all()

    return {
        "label":   label,
        "entries": [
            {
                "text":           r.canonical_form,
                "label":          r.label,
                "case_count":     r.case_count,
                "total_mentions": r.total_mentions or 0,
            }
            for r in rows
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RECENT ACTIVITY FEED
# ══════════════════════════════════════════════════════════════════════════════

async def get_recent_activity(limit: int = 50) -> dict:
    """
    Latest raw_data records across all cases — activity feed.
    """
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(RawData)
                .order_by(RawData.fetched_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    return {
        "records": [
            {
                "id":          str(r.id),
                "case_id":     str(r.case_id),
                "source_name": r.source_name,
                "source_url":  r.source_url[:80] if r.source_url else None,
                "fetched_at":  r.fetched_at.isoformat() if r.fetched_at else None,
                "robots_ok":   r.robots_allowed,
                "content":     r.content,
            }
            for r in rows
        ],
        "total": len(rows),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NEO4J GRAPH STATS
# ══════════════════════════════════════════════════════════════════════════════

async def get_graph_stats() -> dict:
    """
    Node and edge counts from Neo4j identity graph.
    """
    try:
        from prahar.modules.c06_graph.driver import run_query
        from prahar.modules.c06_graph.schema import (
            NODE_IDENTITY, NODE_FRAGMENT, NODE_ENTITY,
            NODE_EVIDENCE, NODE_CASE,
            REL_CONTRADICTS, REL_CORROBORATES, REL_SHARES_PLATFORM,
        )

        node_counts, edge_counts = await asyncio.gather(
            run_query(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
            ),
            run_query(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
            ),
        )

        return {
            "nodes":      {r["label"]: r["cnt"] for r in node_counts if r["label"]},
            "edges":      {r["rel_type"]: r["cnt"] for r in edge_counts if r["rel_type"]},
            "available":  True,
        }
    except Exception as e:
        logger.warning(f"[C-09] Neo4j graph stats unavailable: {e}")
        return {"available": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  CASE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

async def search_cases(query: str, limit: int = 20) -> dict:
    """
    Full-text search over source names, URLs, and seed hashes.
    Returns matching cases with context snippets.
    """
    q = f"%{query.lower()}%"

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(RawData)
                .where(
                    RawData.source_name.ilike(q) |
                    RawData.source_url.ilike(q) |
                    RawData.seed_hash.ilike(q)
                )
                .order_by(RawData.fetched_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    seen_cases = set()
    results = []
    for r in rows:
        cid = str(r.case_id)
        if cid not in seen_cases:
            seen_cases.add(cid)
            results.append({
                "case_id":     cid,
                "source_name": r.source_name,
                "source_url":  r.source_url,
                "seed_hash":   r.seed_hash,
                "fetched_at":  r.fetched_at.isoformat() if r.fetched_at else None,
            })

    return {
        "query":   query,
        "results": results,
        "total":   len(results),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  QUOTA STATUS
# ══════════════════════════════════════════════════════════════════════════════

async def get_quota_status() -> dict:
    """
    Check current API quota usage from Redis.
    """
    try:
        import os, redis.asyncio as aioredis
        from prahar.modules.c01_ingestion.connectors import QUOTAS

        r = await aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        month = datetime.utcnow().strftime("%Y-%m")
        quotas = {}
        for source, cfg in QUOTAS.items():
            key   = f"prahar:quota:{source}:{month}"
            used  = int(await r.get(key) or 0)
            limit = cfg["limit"]
            quotas[source] = {
                "used":    used,
                "limit":   limit,
                "pct":     round(used / limit * 100, 1) if limit else 0,
                "period":  cfg["period"],
                "status":  "ok" if used < limit * 0.8 else
                           "warning" if used < limit else "exhausted",
            }
        await r.aclose()
        return {"quotas": quotas, "available": True}
    except Exception as e:
        logger.warning(f"[C-09] Quota status unavailable: {e}")
        return {"available": False, "error": str(e)}
