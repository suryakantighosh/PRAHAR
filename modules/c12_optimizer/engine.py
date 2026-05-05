"""
prahar/modules/c12_optimizer/engine.py
C-12 AMCE Weight Optimizer — async orchestrator.

Pipeline:
  1. Load all unprocessed feedback_event rows from DB
  2. Join with threat_score to recover the (l1, l2, l3, l4) layer scores
     that produced each predicted score
  3. Build FeedbackRecord list and run gradient descent via optimizer.py
  4. Persist updated AMCE weights to amce_weights table
  5. Persist updated CPIF weights to existing signal_weights table
  6. Return convergence statistics

Also exposes:
  load_current_amce_weights()  → AMCEWeights  (for amce.py to call)
  load_current_cpif_weights()  → CPIFWeights  (for cpif.py to call)
  record_feedback(...)         → persists one analyst correction event
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import Column, DateTime, Float, Integer, String, select, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase

from prahar.core.db import AsyncSessionLocal
from prahar.models.amce import ThreatScore, SignalWeights
from prahar.modules.c12_optimizer.optimizer import (
    AMCEWeights,
    CPIFWeights,
    FeedbackRecord,
    OptimizerState,
    run_until_convergence,
)


# ── ORM models ─────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class FeedbackEvent(_Base):
    """
    Mirrors the feedback_event table defined in init.sql.
    `delta` is a generated column in Postgres — not mapped here to
    avoid SQLAlchemy write conflicts; read via raw query if needed.
    """
    __tablename__ = "feedback_event"

    id              = Column(PG_UUID(as_uuid=True), primary_key=True,
                             default=uuid4)
    case_id         = Column(PG_UUID(as_uuid=True), nullable=False)
    identity_id     = Column(PG_UUID(as_uuid=True))
    predicted_score = Column(Float, nullable=False)
    analyst_score   = Column(Float, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)


class AMCEWeightRecord(_Base):
    """
    Persisted snapshot of AMCE layer weights after each optimizer run.
    Maintains a full history so weights can be rolled back if needed.
    """
    __tablename__ = "amce_weights"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    w_l1       = Column(Float, nullable=False, default=0.35)
    w_l2       = Column(Float, nullable=False, default=0.30)
    w_l3       = Column(Float, nullable=False, default=0.20)
    w_l4       = Column(Float, nullable=False, default=0.15)
    loss       = Column(Float)          # MSE on the feedback batch
    n_feedback = Column(Integer)        # number of events used
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def load_current_amce_weights() -> AMCEWeights:
    """
    Load the most recent AMCE weight record from DB.
    Returns defaults if no record exists yet.
    """
    async with AsyncSessionLocal() as db:
        stmt = (
            select(AMCEWeightRecord)
            .order_by(AMCEWeightRecord.id.desc())
            .limit(1)
        )
        row = (await db.execute(stmt)).scalars().first()

    if row is None:
        logger.info("[C-12] No weight record found — using defaults")
        return AMCEWeights()

    return AMCEWeights(
        w_l1=row.w_l1,
        w_l2=row.w_l2,
        w_l3=row.w_l3,
        w_l4=row.w_l4,
    )


async def load_current_cpif_weights() -> CPIFWeights:
    """
    Load the most recent CPIF signal weights from DB.
    Returns defaults if no record exists yet.
    """
    async with AsyncSessionLocal() as db:
        stmt = (
            select(SignalWeights)
            .order_by(SignalWeights.id.desc())
            .limit(1)
        )
        row = (await db.execute(stmt)).scalars().first()

    if row is None:
        logger.info("[C-12] No signal_weights record — using defaults")
        return CPIFWeights()

    return CPIFWeights(
        w_bio=row.w_bio,
        w_usr=row.w_usr,
        w_tbs=row.w_tbs,
    )


async def record_feedback(
    case_id: UUID,
    identity_id: UUID,
    predicted_score: float,
    analyst_score: float,
) -> UUID:
    """
    Persist one analyst correction event to the feedback_event table.

    Parameters
    ----------
    case_id         : the case this feedback applies to
    identity_id     : the consolidated identity node being corrected
    predicted_score : AMCE score that was shown to the analyst
    analyst_score   : ground-truth score the analyst assigned

    Returns
    -------
    UUID of the created feedback_event row.
    """
    async with AsyncSessionLocal() as db:
        event = FeedbackEvent(
            case_id=case_id,
            identity_id=identity_id,
            predicted_score=float(predicted_score),
            analyst_score=float(analyst_score),
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        eid = event.id

    logger.info(
        f"[C-12] Feedback recorded id={eid} "
        f"predicted={predicted_score:.3f} analyst={analyst_score:.3f} "
        f"delta={analyst_score - predicted_score:+.3f}"
    )
    return eid


async def _load_feedback_records(limit: int = 500) -> list[FeedbackRecord]:
    """
    Load recent feedback events joined with threat_score to recover
    per-layer scores (l1–l4) needed for the gradient computation.

    Uses the most recent `limit` events ordered by creation time.
    """
    records: list[FeedbackRecord] = []

    async with AsyncSessionLocal() as db:
        # Load feedback events
        fb_stmt = (
            select(FeedbackEvent)
            .order_by(FeedbackEvent.created_at.desc())
            .limit(limit)
        )
        fb_rows = (await db.execute(fb_stmt)).scalars().all()

        if not fb_rows:
            return []

        # For each feedback event, find the corresponding threat_score row
        # matched by (case_id, identity_id) — take the most recent score
        # that was created before the feedback event.
        for fb in fb_rows:
            ts_stmt = (
                select(ThreatScore)
                .where(ThreatScore.case_id == fb.case_id)
                .where(ThreatScore.identity_id == fb.identity_id)
                .order_by(ThreatScore.created_at.desc())
                .limit(1)
            )
            ts_row = (await db.execute(ts_stmt)).scalars().first()

            if ts_row is None:
                # No matching threat_score — can't compute gradient, skip
                logger.debug(
                    f"[C-12] No threat_score for feedback {fb.id} "
                    f"case={fb.case_id} — skipping"
                )
                continue

            records.append(FeedbackRecord(
                predicted_score=fb.predicted_score,
                analyst_score=fb.analyst_score,
                l1=ts_row.score_l1 or 0.0,
                l2=ts_row.score_l2 or 0.0,
                l3=ts_row.score_l3 or 0.0,
                l4=ts_row.score_l4 or 0.0,
                # CPIF component scores not stored in DB yet — omit for now
                cpif_bio=None,
                cpif_usr=None,
                cpif_tbs=None,
            ))

    logger.info(
        f"[C-12] Loaded {len(records)} feedback records "
        f"(of {len(fb_rows)} events, {len(fb_rows) - len(records)} skipped)"
    )
    return records


async def _persist_amce_weights(
    weights: AMCEWeights,
    loss: float,
    n_feedback: int,
) -> None:
    """Write an AMCEWeightRecord snapshot to DB."""
    async with AsyncSessionLocal() as db:
        row = AMCEWeightRecord(
            w_l1=weights.w_l1,
            w_l2=weights.w_l2,
            w_l3=weights.w_l3,
            w_l4=weights.w_l4,
            loss=loss,
            n_feedback=n_feedback,
        )
        db.add(row)
        await db.commit()

    logger.success(
        f"[C-12] AMCE weights persisted: "
        f"L1={weights.w_l1:.4f} L2={weights.w_l2:.4f} "
        f"L3={weights.w_l3:.4f} L4={weights.w_l4:.4f} "
        f"loss={loss:.6f} n={n_feedback}"
    )


async def _persist_cpif_weights(weights: CPIFWeights) -> None:
    """Insert a new signal_weights row (table keeps full history)."""
    async with AsyncSessionLocal() as db:
        row = SignalWeights(
            w_bio=weights.w_bio,
            w_usr=weights.w_usr,
            w_tbs=weights.w_tbs,
        )
        db.add(row)
        await db.commit()

    logger.success(
        f"[C-12] CPIF weights persisted: "
        f"bio={weights.w_bio:.4f} usr={weights.w_usr:.4f} "
        f"tbs={weights.w_tbs:.4f}"
    )


# ── Main engine function ───────────────────────────────────────────────────────

async def run_optimizer(
    lr: float = 0.01,
    max_epochs: int = 200,
    patience: int = 10,
    feedback_limit: int = 500,
) -> dict:
    """
    Full optimizer pipeline:
      1. Load current weights from DB
      2. Load feedback events (joined with threat_score)
      3. Run gradient descent to convergence
      4. Persist updated weights
      5. Return stats

    Returns
    -------
    {
      "feedback_used":   int,
      "epochs_run":      int,
      "loss_before":     float,
      "loss_after":      float,
      "converged":       bool,
      "amce_weights":    dict,
      "cpif_weights":    dict,
    }
    """
    logger.info("[C-12] Starting AMCE weight optimization run")

    # ── 1. Load current weights ───────────────────────────────────────────────
    amce_w, cpif_w = await asyncio.gather(
        load_current_amce_weights(),
        load_current_cpif_weights(),
    )

    # ── 2. Load feedback records ──────────────────────────────────────────────
    records = await _load_feedback_records(limit=feedback_limit)

    if not records:
        logger.warning("[C-12] No usable feedback records — weights unchanged")
        return {
            "feedback_used": 0,
            "epochs_run": 0,
            "loss_before": 0.0,
            "loss_after": 0.0,
            "converged": True,
            "amce_weights": vars(amce_w),
            "cpif_weights": vars(cpif_w),
        }

    # ── 3. Gradient descent ───────────────────────────────────────────────────
    state = OptimizerState(amce=amce_w, cpif=cpif_w)
    loop  = asyncio.get_event_loop()

    stats = await loop.run_in_executor(
        None,
        lambda: run_until_convergence(
            records, state,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
        )
    )

    # ── 4. Persist ────────────────────────────────────────────────────────────
    await asyncio.gather(
        _persist_amce_weights(
            state.amce,
            loss=stats["loss_after"],
            n_feedback=len(records),
        ),
        _persist_cpif_weights(state.cpif),
    )

    logger.info(
        f"[C-12] Optimization complete: "
        f"epochs={stats['total_epochs']} "
        f"loss {stats['loss_before']:.6f} → {stats['loss_after']:.6f} "
        f"converged={stats['converged']}"
    )

    return {
        "feedback_used":  len(records),
        "epochs_run":     stats["total_epochs"],
        "loss_before":    stats["loss_before"],
        "loss_after":     stats["loss_after"],
        "converged":      stats["converged"],
        "amce_weights":   vars(state.amce),
        "cpif_weights":   vars(state.cpif),
    }


async def get_weight_history(limit: int = 20) -> list[dict]:
    """
    Return the most recent AMCE weight snapshots for monitoring.
    Useful for the c09 dashboard.
    """
    async with AsyncSessionLocal() as db:
        stmt = (
            select(AMCEWeightRecord)
            .order_by(AMCEWeightRecord.id.desc())
            .limit(limit)
        )
        rows = (await db.execute(stmt)).scalars().all()

    return [
        {
            "id":         row.id,
            "w_l1":       row.w_l1,
            "w_l2":       row.w_l2,
            "w_l3":       row.w_l3,
            "w_l4":       row.w_l4,
            "loss":       row.loss,
            "n_feedback": row.n_feedback,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
        for row in reversed(rows)   # chronological order
    ]
