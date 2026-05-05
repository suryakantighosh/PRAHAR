"""
prahar/models/nlp.py
ORM models for C-05 NLP Pipeline
"""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Entity(Base):
    __tablename__ = "entity"

    entity_id      = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id        = Column(UUID(as_uuid=True), nullable=False, index=True)
    text           = Column(Text, nullable=False)
    label          = Column(Text, nullable=False)   # PERSON, ORG, GPE, DATE...
    canonical_form = Column(Text)                   # after dedup/coref
    raw_data_ref   = Column(UUID(as_uuid=True))     # → raw_data.id
    meta           = Column(JSONB)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class EntityAlias(Base):
    __tablename__ = "entity_alias"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_id   = Column(UUID(as_uuid=True), nullable=False, index=True)
    alias_text  = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
