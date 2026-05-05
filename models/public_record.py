"""
prahar/models/public_record.py
ORM models for C-04 Public Records Crawler
"""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PublicRecord(Base):
    __tablename__ = "public_record"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id     = Column(UUID(as_uuid=True), nullable=False, index=True)
    record_type = Column(Text, nullable=False)   # MCA21 / ECOURT / GAZETTE / NEWS
    source      = Column(Text, nullable=False)
    subject     = Column(Text)                   # name / CIN / case number
    content     = Column(JSONB)
    source_url  = Column(Text)
    record_date = Column(Text)                   # keep as string — gazette dates vary
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)


class NewsRecord(Base):
    __tablename__ = "news_record"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id     = Column(UUID(as_uuid=True), nullable=False, index=True)
    title       = Column(Text)
    snippet     = Column(Text)
    source_url  = Column(Text)
    publisher   = Column(Text)
    published   = Column(Text)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
