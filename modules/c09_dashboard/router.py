"""
prahar/modules/c09_dashboard/router.py
C-09 Dashboard — FastAPI router.

Mount this on the main FastAPI app:
    from prahar.modules.c09_dashboard.router import router as dashboard_router
    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["dashboard"])

Endpoints:
  GET  /health                           -> system health check
  GET  /stats                            -> pipeline record counts
  GET  /cases                            -> paginated case list
  GET  /cases/{case_id}                  -> case detail
  GET  /cases/{case_id}/brief/pdf        -> download intelligence PDF
  GET  /cases/{case_id}/brief/stix       -> download STIX 2.1 bundle
  POST /cases/{case_id}/feedback         -> submit analyst score correction
  GET  /weights                          -> AMCE/CPIF weight dashboard
  POST /weights/optimize                 -> trigger weight optimization run
  GET  /entities                         -> entity leaderboard
  GET  /activity                         -> recent activity feed
  GET  /graph                            -> Neo4j graph stats
  GET  /search                           -> full-text case search
  GET  /quotas                           -> API quota status
"""

from __future__ import annotations

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
from loguru import logger
from prahar.modules.c09_dashboard.engine import AsyncSessionLocal
from prahar.modules.c09_dashboard.engine import (
    get_system_health,
    get_pipeline_stats,
    get_case_list,
    get_case_detail,
    get_weight_dashboard,
    get_entity_leaderboard,
    get_recent_activity,
    get_graph_stats,
    search_cases,
    get_quota_status,
)

router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", summary="System health check")
async def health():
    """Ping all infrastructure services and return status."""
    return await get_system_health()



# At the top of router.py (or in engine.py)
async def _resolve_subject_name(case_id, detail) -> str:
    from prahar.modules.c09_dashboard.engine import AsyncSessionLocal
    from prahar.models.raw_data import RawData as _RD
    from sqlalchemy import select as _sel
    usernames = detail.get("identity", {}).get("usernames") or []
    name = ", ".join(usernames[:2])
    if not name:
        async with AsyncSessionLocal() as db:
            raws = (await db.execute(_sel(_RD).where(_RD.case_id == case_id).limit(15))).scalars().all()
            for r in raws:
                for key in ("name","title","display_name","full_name"):
                    v = (r.content or {}).get(key, "")
                    if v and isinstance(v, str) and 2 < len(v) < 80 and " " in v:
                        name = v; break
                if name: break
    return name or str(case_id)[:8]
# ── Pipeline stats ─────────────────────────────────────────────────────────────

@router.get("/stats", summary="Pipeline record counts")
async def pipeline_stats():
    """Row counts per module — total and today's ingestion."""
    return await get_pipeline_stats()


# ── Cases ──────────────────────────────────────────────────────────────────────

@router.get("/cases", summary="Paginated case list")
async def case_list(
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by:   str = Query("last_activity", regex="^(last_activity|risk_score|created)$"),
):
    """List all investigation cases with summary statistics."""
    return await get_case_list(page=page, page_size=page_size, sort_by=sort_by)


@router.get("/cases/{case_id}", summary="Case detail report")
async def case_detail(case_id: UUID):
    """
    Full report card for one case — identity fragments, entities,
    news, public records, threat scores, and breach data.
    """
    try:
        return await get_case_detail(case_id)
    except Exception as e:
        logger.error(f"[C-09] case_detail error case={case_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/cases/{case_id}/brief/pdf",
    summary="Download intelligence brief as PDF",
    response_class=Response,
)
async def case_brief_pdf(case_id: UUID):
    """Generate and stream a ReportLab PDF intelligence brief for this case."""
    try:
        from datetime import datetime as dt
        from prahar.modules.c08_brief.pdf_builder import build_pdf
        from prahar.modules.c08_brief.phc import build_provenance_chain, chain_to_dict

        detail = await get_case_detail(case_id)
        # Smart subject name extraction from raw data content
        from prahar.models.raw_data import RawData as _RawData2
        from sqlalchemy import select as _sel2
        _subj_name = await _resolve_subject_name(case_id, detail)
        if not _subj_name:
            async with AsyncSessionLocal() as _db2:
                _raws = (await _db2.execute(_sel2(_RawData2).where(_RawData2.case_id==case_id).limit(15))).scalars().all()
                for _rr in _raws:
                    _cc = _rr.content or {}
                    # Try multiple fields where name could be stored
                    for _key in ("name","title","display_name","displayName","full_name","preferredUsername","subject"):
                        _nv = _cc.get(_key,"")
                        if _nv and isinstance(_nv,str) and 2<len(_nv)<80 and not _nv.startswith("http") and " " in _nv:
                            _subj_name = _nv; break
                    if _subj_name: break
            if not _subj_name: _subj_name = detail["case_id"][:8]

        prov_nodes = [
            {"node_id": detail["case_id"], "node_type": "case",    "content": detail["summary"]},
            {"node_id": detail["case_id"], "node_type": "risk",    "content": detail["risk"]},
            {"node_id": detail["case_id"], "node_type": "identity","content": detail["identity"]},
        ]
        chain = build_provenance_chain(prov_nodes)
        prov_hash = chain[-1].chain_hash if chain else "0" * 64

        pdf_bytes = build_pdf(
            case_id=detail["case_id"],
            subject_name=_subj_name,
            generated_at=dt.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            risk_level=detail["risk"]["level"],
            final_score=detail.get("risk", {}).get("score") or 0.0,
            risk_flags=detail["risk"]["flags"],
            platforms=detail["identity"]["platforms"],
            breach_names=detail["identity"]["breach_names"],
            top_persons=[e.get("text") for e in detail.get("entities", {}).get("top_persons", [])],
            top_orgs=[e.get("text") for e in detail.get("entities", {}).get("top_persons", [])],
            amce_breakdown={"contributions": {
                "l1_weighted": detail["risk"]["score_l1"],
                "l2_weighted": detail["risk"]["score_l2"],
                "l3_weighted": detail["risk"]["score_l3"],
                "l4_penalty":  detail["risk"]["score_l4"],
            }},
            provenance_chain=chain_to_dict(chain),
            provenance_hash=prov_hash,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f'attachment; filename="prahar_brief_{str(case_id)[:8]}.pdf"'
            },
        )
    except Exception as e:
        logger.error(f"[C-09] PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/cases/{case_id}/brief/stix",
    summary="Download STIX 2.1 JSON bundle",
    response_class=Response,
)
async def case_brief_stix(case_id: UUID):
    """Generate a STIX 2.1 JSON bundle for this case."""
    try:
        from prahar.modules.c08_brief.stix_builder import brief_to_stix

        detail = await get_case_detail(case_id)
        _subj_name = await _resolve_subject_name(case_id, detail)
        stix_json = brief_to_stix(
            case_id=detail["case_id"],
            subject_name=_subj_name,
            platforms=detail["identity"]["platforms"],
            risk_level=detail["risk"]["level"],
            confidence_score=detail["risk"]["score"] or 0.0,
            risk_flags=detail["risk"]["flags"],
            breach_names=detail["identity"]["breach_names"],
            provenance_hash="0" * 64,
        )

        return Response(
            content=stix_json,
            media_type="application/json",
            headers={
                "Content-Disposition":
                    f'attachment; filename="prahar_stix_{str(case_id)[:8]}.json"'
            },
        )
    except Exception as e:
        logger.error(f"[C-09] STIX generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class FeedbackPayload(BaseModel):
    identity_id:     UUID
    predicted_score: float = Field(..., ge=0.0, le=1.0)
    analyst_score:   float = Field(..., ge=0.0, le=1.0)
    notes:           Optional[str] = None


@router.post("/cases/{case_id}/feedback", summary="Submit analyst score correction")
async def submit_feedback(case_id: UUID, payload: FeedbackPayload):
    """Record an analyst correction to feed C-12 optimizer."""
    try:
        from prahar.modules.c12_optimizer.engine import record_feedback

        event_id = await record_feedback(
            case_id=case_id,
            identity_id=payload.identity_id,
            predicted_score=payload.predicted_score,
            analyst_score=payload.analyst_score,
        )
        return {
            "status":   "recorded",
            "event_id": str(event_id),
            "delta":    round(payload.analyst_score - payload.predicted_score, 4),
        }
    except Exception as e:
        logger.error(f"[C-09] Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Weights ────────────────────────────────────────────────────────────────────

@router.get("/weights", summary="AMCE/CPIF weight dashboard")
async def weight_dashboard():
    """Current weights, historical trajectories, and loss curve."""
    return await get_weight_dashboard()


@router.post("/weights/optimize", summary="Trigger weight optimization run")
async def trigger_optimization(background_tasks: BackgroundTasks):
    """Launch a C-12 gradient descent run in the background."""
    from prahar.modules.c12_optimizer.engine import run_optimizer

    async def _run():
        try:
            stats = await run_optimizer()
            logger.info(f"[C-09] Background optimization complete: {stats}")
        except Exception as e:
            logger.error(f"[C-09] Background optimization failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "optimization_started", "message": "Check /weights for updated values."}


# ── Entity leaderboard ─────────────────────────────────────────────────────────

@router.get("/entities", summary="Entity leaderboard across all cases")
async def entity_leaderboard(
    label: str = Query("PERSON", regex="^(PERSON|ORG|GPE|LOC|NORP|DATE|MONEY)$"),
    limit: int = Query(20, ge=1, le=100),
):
    """Top entities by case frequency and total mentions."""
    return await get_entity_leaderboard(label=label, limit=limit)


# ── Activity feed ──────────────────────────────────────────────────────────────

@router.get("/activity", summary="Recent activity feed")
async def recent_activity(limit: int = Query(50, ge=1, le=200)):
    """Latest ingested records across all active cases."""
    return await get_recent_activity(limit=limit)


# ── Graph stats ────────────────────────────────────────────────────────────────

@router.get("/graph", summary="Neo4j identity graph statistics")
async def graph_stats():
    """Node and edge counts from the identity evidence graph."""
    return await get_graph_stats()


# ── Search ─────────────────────────────────────────────────────────────────────

@router.get("/search", summary="Full-text case search")
async def case_search(
    q:     str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=100),
):
    """Search cases by source name, URL, or seed hash."""
    return await search_cases(query=q, limit=limit)


# ── Quotas ─────────────────────────────────────────────────────────────────────

@router.get("/quotas", summary="API quota status")
async def quota_status():
    """Current API quota usage for all rate-limited sources."""
    return await get_quota_status()
