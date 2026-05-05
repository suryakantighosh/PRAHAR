"""
prahar/models/amce.py
ORM models for C-07 AMCE Scoring
"""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, Float, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ThreatScore(Base):
    __tablename__ = "threat_score"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id     = Column(UUID(as_uuid=True), nullable=False, index=True)
    identity_id = Column(UUID(as_uuid=True))
    score_l1    = Column(Float)   # raw signal scoring
    score_l2    = Column(Float)   # structural corroboration
    score_l3    = Column(Float)   # behavioral alignment
    score_l4    = Column(Float)   # conflict penalty
    final_score = Column(Float)
    risk_flags  = Column(ARRAY(Text))
    risk_level  = Column(Text, default='UNSCORED')
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)


class SignalWeights(Base):
    __tablename__ = "signal_weights"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    w_bio       = Column(Float, default=0.40, nullable=False)
    w_usr       = Column(Float, default=0.35, nullable=False)
    w_tbs       = Column(Float, default=0.25, nullable=False)
    updated_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
