from dotenv import load_dotenv
load_dotenv()
"""
prahar/api/main.py
PRAHAR v2 — FastAPI application entry point.

Start with:
    uvicorn prahar.api.main:app --host 0.0.0.0 --port 8000 --reload

Or via Docker:
    docker-compose up prahar-api
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from prahar.modules.c09_dashboard.router import router as dashboard_router


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PRAHAR v2 API starting up…")
    # Ensure Neo4j indexes exist
    try:
        from prahar.modules.c06_graph.driver import ensure_indexes
        await ensure_indexes()
        logger.info("[startup] Neo4j indexes ensured")
    except Exception as e:
        logger.warning(f"[startup] Neo4j not available: {e}")
    yield
    logger.info("PRAHAR v2 API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PRAHAR v2 — OSINT Intelligence Platform",
    description=(
        "Cross-Platform Identity Fusion (CPIF) and Adaptive Multi-Layer "
        "Confidence Engine (AMCE) powered OSINT analysis API."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(
    dashboard_router,
    prefix="/api/v1/dashboard",
    tags=["C-09 Dashboard"],
)


# ── Ingestion task endpoints ──────────────────────────────────────────────────

from fastapi import BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class IngestDomainRequest(BaseModel):
    domain:  str
    case_id: Optional[UUID] = None


class IngestUsernameRequest(BaseModel):
    username: str
    case_id:  Optional[UUID] = None


class IngestPersonRequest(BaseModel):
    name:      str
    username:  Optional[str] = ""
    email:     Optional[str] = ""
    phone:     Optional[str] = ""
    image_b64: Optional[str] = ""
    case_id:   Optional[UUID] = None
class ResolveUsernameRequest(BaseModel):
    username: str
    case_id:  Optional[UUID] = None


class ScoreCaseRequest(BaseModel):
    case_id:         UUID
    identity_id:     Optional[UUID] = None
    source_count:    int = 0
    platform_count:  int = 0
    breach_count:    int = 0
    graph_degree:    int = 0
    corroboration_score: float = 0.0
    cpif_score:      float = 0.0
    sif_similarity:  float = 0.5
    tbs_kl_score:    float = 0.5
    conflict_count:  int = 0
    conflict_weight: float = 0.0


@app.post("/api/v1/ingest/domain", tags=["C-01 Ingestion"])
async def ingest_domain(req: IngestDomainRequest, background_tasks: BackgroundTasks):
    """Trigger domain ingestion pipeline (C-01)."""
    from prahar.modules.c01_ingestion.engine import ingest_domain

    async def _run():
        try:
            result = await ingest_domain(req.domain, req.case_id)
            logger.success(f"[API] Domain ingestion complete: {result}")
        except Exception as e:
            logger.error(f"[API] Domain ingestion error: {e}")

    background_tasks.add_task(_run)
    return {"status": "ingestion_started", "domain": req.domain}


@app.post("/api/v1/ingest/username", tags=["C-01 Ingestion"])
async def ingest_username(req: IngestUsernameRequest, background_tasks: BackgroundTasks):
    """Trigger username ingestion pipeline (C-01)."""
    from prahar.modules.c01_ingestion.engine import ingest_username

    async def _run():
        try:
            result = await ingest_username(req.username, req.case_id)
            logger.success(f"[API] Username ingestion complete: {result}")
        except Exception as e:
            logger.error(f"[API] Username ingestion error: {e}")

    background_tasks.add_task(_run)
    return {"status": "ingestion_started", "username": req.username}


@app.post("/api/v1/resolve/username", tags=["C-02 Identity"])
async def resolve_username(req: ResolveUsernameRequest, background_tasks: BackgroundTasks):
    """Resolve username through CPIF identity fusion pipeline (C-02)."""
    from prahar.modules.c02_identity.engine import resolve_username

    async def _run():
        try:
            result = await resolve_username(req.username, req.case_id)
            logger.success(f"[API] Username resolved: {result}")
        except Exception as e:
            logger.error(f"[API] Username resolve error: {e}")

    background_tasks.add_task(_run)
    return {"status": "resolution_started", "username": req.username}


@app.post("/api/v1/score/case", tags=["C-07 AMCE"])
async def score_case(req: ScoreCaseRequest):
    """Compute AMCE threat score for a case (C-07)."""
    from prahar.modules.c07_amce.amce import AMCEInput, compute_amce
    from prahar.modules.c12_optimizer.engine import load_current_amce_weights
    from prahar.modules.c09_dashboard.engine import AsyncSessionLocal
    from prahar.models.amce import ThreatScore

    weights = await load_current_amce_weights()

    inp = AMCEInput(
        identity_id=str(req.identity_id or req.case_id),
        source_count=req.source_count,
        platform_count=req.platform_count,
        breach_count=req.breach_count,
        graph_degree=req.graph_degree,
        corroboration_score=req.corroboration_score,
        cpif_score=req.cpif_score,
        sif_similarity=req.sif_similarity,
        tbs_kl_score=req.tbs_kl_score,
        conflict_count=req.conflict_count,
        conflict_weight=req.conflict_weight,
    )

    result = compute_amce(
        inp,
        w_l1=weights.w_l1,
        w_l2=weights.w_l2,
        w_l3=weights.w_l3,
        w_l4=weights.w_l4,
    )

    # Persist to DB
    async with AsyncSessionLocal() as db:
        ts = ThreatScore(
            case_id=req.case_id,
            identity_id=req.identity_id,
            score_l1=result.score_l1,
            score_l2=result.score_l2,
            score_l3=result.score_l3,
            score_l4=result.score_l4,
            final_score=result.final_score,
            risk_flags=result.risk_flags,
        )
        db.add(ts)
        await db.commit()
        await db.refresh(ts)

    return {
        "score_id":    str(ts.id),
        "case_id":     str(req.case_id),
        "final_score": result.final_score,
        "risk_level":  result.risk_level,
        "risk_flags":  result.risk_flags,
        "breakdown":   result.breakdown,
    }


@app.post("/api/v1/nlp/process/{case_id}", tags=["C-05 NLP"])
async def process_nlp(case_id: UUID, background_tasks: BackgroundTasks):
    """Run NLP entity extraction on all raw_data for a case (C-05)."""
    from prahar.modules.c05_nlp.engine import process_case_text

    async def _run():
        try:
            result = await process_case_text(case_id)
            logger.success(f"[API] NLP processing complete: {result}")
        except Exception as e:
            logger.error(f"[API] NLP processing error: {e}")

    background_tasks.add_task(_run)
    return {"status": "nlp_started", "case_id": str(case_id)}


@app.post("/api/v1/tbs/compute/{case_id}", tags=["C-11 TBS"])
async def compute_tbs(case_id: UUID, background_tasks: BackgroundTasks):
    """Compute Temporal Behavioral Profile for a case (C-11)."""
    from prahar.modules.c11_tbs.engine import compute_tbp_for_case

    async def _run():
        try:
            result = await compute_tbp_for_case(case_id)
            logger.success(f"[API] TBS profile computed: {result}")
        except Exception as e:
            logger.error(f"[API] TBS error: {e}")

    background_tasks.add_task(_run)
    return {"status": "tbs_started", "case_id": str(case_id)}


@app.post("/api/v1/sif/compute/{case_id}", tags=["C-10 SIF"])
async def compute_sif(case_id: UUID, background_tasks: BackgroundTasks):
    """Compute Stylometric Identity Fingerprint for a case (C-10)."""
    from prahar.modules.c10_sif.engine import compute_sfv_for_case

    async def _run():
        try:
            result = await compute_sfv_for_case(case_id)
            logger.success(f"[API] SIF computed: {result}")
        except Exception as e:
            logger.error(f"[API] SIF error: {e}")

    background_tasks.add_task(_run)
    return {"status": "sif_started", "case_id": str(case_id)}

@app.post("/api/v1/ingest/person", tags=["C-01 Ingestion"])
async def ingest_person(req: IngestPersonRequest, background_tasks: BackgroundTasks):
    """
    Full OSINT person investigation.
    Pre-generates case_id and returns it immediately so GUI can track progress.
    """
    from prahar.modules.c01_ingestion.engine import ingest_person
    from prahar.modules.c01_ingestion.seed import make_case_id

    # Pre-generate case_id so GUI knows it immediately
    pre_case_id = req.case_id or make_case_id()

    async def _run():
        try:
            result = await ingest_person(
                name=req.name,
                username=req.username or "",
                email=req.email or "",
                phone=req.phone or "",
                image_b64=req.image_b64 or "",
                case_id=pre_case_id,
            )
            logger.success(f"[API] Person OSINT complete: {result}")
        except Exception as e:
            logger.error(f"[API] Person OSINT error: {e}")

    background_tasks.add_task(_run)
    return {
        "status":    "osint_started",
        "case_id":   str(pre_case_id),
        "name":      req.name,
        "username":  req.username,
        "email":     req.email,
        "phone":     req.phone,
        "has_photo": bool(req.image_b64),
    }

# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "name":    "PRAHAR v2",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
        "health":  "/api/v1/dashboard/health",
    })
