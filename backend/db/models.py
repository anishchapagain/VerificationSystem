"""
SQLAlchemy ORM Models
=======================
Module  : db/models.py
Purpose : Define the relational schema for the Signature Verifier application.
          Three core tables: users, signatures, match_logs.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Integer, LargeBinary, String, Text, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


class User(Base):
    """
    Represents an application user who owns one or more reference signatures.

    Attributes:
        id         : Auto-incrementing primary key.
        name       : Full display name.
        email      : Unique email address used as login identifier.
        is_active  : Soft-delete flag (False = account disabled).
        created_at : UTC timestamp of account creation.
        signatures : One-to-many relationship to Signature records.
        match_logs : One-to-many relationship to MatchLog records.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    signatures: Mapped[List["Signature"]] = relationship(
        "Signature", back_populates="user", cascade="all, delete-orphan"
    )
    match_logs: Mapped[List["MatchLog"]] = relationship(
        "MatchLog", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} active={self.is_active}>"


class Signature(Base):
    """
    A stored reference signature belonging to a specific user.

    Each user may have multiple reference signatures (e.g., captured on
    different days) to improve matching robustness.

    Attributes:
        id           : Auto-incrementing primary key.
        user_id      : FK to the owning User.
        label        : Optional descriptive label (e.g. "Primary", "Backup").
        file_path    : Relative path to the stored preprocessed image.
        embedding    : Raw bytes of the serialized 512-D numpy float32 array.
        faiss_id     : Integer row index in the FAISS flat index.
        created_at   : UTC timestamp of upload.
        user         : Back-reference to the owning User.
        match_logs   : Logs where this signature was the best match.
    """

    __tablename__ = "signatures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False,
        comment="Serialized numpy float32 array of shape (EMBEDDING_DIM,)"
    )
    faiss_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="signatures")
    match_logs: Mapped[List["MatchLog"]] = relationship(
        "MatchLog", back_populates="best_match", foreign_keys="MatchLog.best_match_id"
    )

    def __repr__(self) -> str:
        return f"<Signature id={self.id} user_id={self.user_id} faiss_id={self.faiss_id}>"


class MatchLog(Base):
    """
    Immutable audit log of every signature verification event.

    Records who was verified, which reference signature was the closest match,
    the raw similarity score, and the final verdict. Used for compliance
    reporting and threshold tuning.

    Attributes:
        id             : Auto-incrementing primary key.
        user_id        : FK to the user being verified.
        best_match_id  : FK to the Signature that scored highest (nullable if none).
        query_path     : Path to the uploaded query image (preserved for audit).
        score          : Cosine similarity score in [0.0, 1.0].
        threshold_used : Threshold value at time of match (for retrospective tuning).
        verdict        : True = MATCH, False = NO MATCH.
        source         : "image" or "video" — the input modality.
        created_at     : UTC timestamp of the verification event.
    """

    __tablename__ = "match_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    best_match_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("signatures.id", ondelete="SET NULL"), nullable=True
    )
    query_path: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="image", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="match_logs")
    best_match: Mapped[Optional["Signature"]] = relationship(
        "Signature", back_populates="match_logs", foreign_keys=[best_match_id]
    )

    def __repr__(self) -> str:
        verdict_str = "MATCH" if self.verdict else "NO MATCH"
        return f"<MatchLog id={self.id} user_id={self.user_id} score={self.score:.4f} verdict={verdict_str}>"
