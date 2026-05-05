"""
prahar/models/identity.py
ORM models for C-02 Identity Resolver + CPIF
"""
from datetime import datetime
from uuid import uuid4
from typing import Optional, List
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class IdentityFragment(Base):
    __tablename__ = "identity_fragment"

    fragment_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id       = Column(UUID(as_uuid=True), nullable=False, index=True)
    platform      = Column(Text, nullable=False)
    username      = Column(Text)
    email         = Column(Text)
    phone         = Column(Text)
    biometric_ref = Column(UUID(as_uuid=True))   # → face_embedding.id
    arf_ref       = Column(UUID(as_uuid=True))   # → C-11 ARF
    sfv_ref       = Column(UUID(as_uuid=True))   # → C-10 SFV
    uncertainty   = Column(Float, default=1.0, nullable=False)
    meta          = Column(JSONB)                # bio, location, follower_count etc.
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)


class ConsolidatedIdentity(Base):
    __tablename__ = "consolidated_identity"

    identity_id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id           = Column(UUID(as_uuid=True), nullable=False, index=True)
    cpif_score        = Column(Float, nullable=False)
    signals_used      = Column(ARRAY(Text))
    analyst_validated = Column(Boolean, default=False)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False)


class BreachRecord(Base):
    __tablename__ = "breach_record"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id      = Column(UUID(as_uuid=True), nullable=False, index=True)
    email        = Column(Text, nullable=False)
    breach_name  = Column(Text)
    breach_date  = Column(Text)
    data_classes = Column(ARRAY(Text))
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


class PhoneRecord(Base):
    __tablename__ = "phone_record"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id   = Column(UUID(as_uuid=True), nullable=False, index=True)
    number    = Column(Text, nullable=False)
    carrier   = Column(Text)
    region    = Column(Text)
    num_type  = Column(Text)   # MOBILE / FIXED_LINE / VOIP etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
