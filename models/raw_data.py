"""
prahar/models/raw_data.py
SQLAlchemy ORM models for C-01 ingestion layer
"""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    Column, String, Boolean, DateTime, JSON, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class RawData(Base):
    __tablename__ = "raw_data"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id      = Column(UUID(as_uuid=True), nullable=False, index=True)
    seed_hash    = Column(Text, nullable=False, index=True)
    source_url   = Column(Text, nullable=False)
    source_name  = Column(Text, nullable=False)
    content      = Column(JSON)
    content_hash = Column(Text, nullable=False)   # SHA-256
    fetched_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    robots_allowed = Column(Boolean, default=True)

    def __repr__(self):
        return f"<RawData case={self.case_id} src={self.source_name}>"
