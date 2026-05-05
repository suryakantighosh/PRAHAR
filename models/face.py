"""
prahar/models/face.py
ORM models for C-03 Face Engine
"""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, Float, Boolean, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class FaceEmbedding(Base):
    __tablename__ = "face_embedding"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id             = Column(UUID(as_uuid=True), nullable=False, index=True)
    source_url          = Column(Text, nullable=False)
    embedding_arcface   = Column(Vector(512))   # DeepFace ArcFace
    embedding_insight   = Column(Vector(512))   # InsightFace buffalo_l
    embedding_openface  = Column(Vector(128))   # dlib/OpenFace
    matched_to          = Column(UUID(as_uuid=True))
    exif_meta           = Column(JSONB)         # date, GPS, camera
    blur_score          = Column(Float)         # reject if < 40
    created_at          = Column(DateTime, default=datetime.utcnow, nullable=False)


class FaceMatch(Base):
    __tablename__ = "face_match"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id          = Column(UUID(as_uuid=True), nullable=False, index=True)
    source_a         = Column(UUID(as_uuid=True), nullable=False)
    source_b         = Column(UUID(as_uuid=True), nullable=False)
    similarity_score = Column(Float, nullable=False)
    consensus_count  = Column(Integer, default=0)   # how many models agreed
    confirmed        = Column(Boolean, default=False)
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)
