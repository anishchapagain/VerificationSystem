"""
SQLAlchemy ORM models for the Signature Verification System.

Three core tables:
    - User         : Registered application users
    - Signature    : Reference signatures stored per user
    - MatchLog     : Audit trail of every verification attempt

All models include created_at / updated_at timestamps and soft-delete support.

Author: Signature Verifier Team
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to any model.

    updated_at is automatically refreshed on every UPDATE via onupdate=func.now().
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    Mixin that adds soft-delete support (is_deleted flag + deleted_at timestamp).

    Rows are never physically removed; they are marked deleted instead,
    preserving the audit trail for compliance and forensic purposes.
    """

    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class User(TimestampMixin, SoftDeleteMixin, Base):
    """
    Represents a registered user of the signature verification system.

    Each user can have multiple reference signatures stored against their account.
    Authentication is handled separately (JWT); this table holds profile data.

    Attributes:
        id          : Auto-increment primary key.
        name        : Full display name.
        email       : Unique login email address.
        hashed_password : Bcrypt-hashed password string.
        is_active   : False = account suspended.
        signatures  : One-to-many relationship to Signature rows.
        match_logs  : One-to-many relationship to MatchLog rows.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    organization: Mapped[str | None] = mapped_column(String(150), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)

    # Relationships
    signatures: Mapped[list["Signature"]] = relationship(
        "Signature",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    match_logs: Mapped[list["MatchLog"]] = relationship(
        "MatchLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} active={self.is_active}>"


class Signature(TimestampMixin, SoftDeleteMixin, Base):
    """
    A reference signature image registered by a user.

    Each Signature row stores the file path of the processed image,
    the L2-normalized 512-dimensional embedding vector (as bytes),
    and the FAISS index position for fast vector retrieval.

    Attributes:
        id              : Auto-increment primary key.
        user_id         : FK to the owning User.
        label           : Optional human-readable label (e.g. "Passport signature").
        file_path       : Absolute path to the stored preprocessed image.
        original_path   : Absolute path to the original uploaded file.
        embedding       : 512-D float32 vector serialized as bytes (numpy tobytes).
        faiss_index_id  : Position of this embedding in the FAISS flat index.
        file_type       : MIME type of the source file (image/png etc.).
        source_type     : 'image' or 'video' — how the signature was captured.
        quality_score   : Laplacian sharpness score of the best frame used.
    """

    __tablename__ = "signatures"
    __table_args__ = (
        UniqueConstraint("user_id", "label", name="uq_user_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False, default="default")
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    faiss_index_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), default="image", nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="signatures")
    match_logs_as_best: Mapped[list["MatchLog"]] = relationship(
        "MatchLog",
        back_populates="best_match_signature",
        foreign_keys="MatchLog.best_match_id",
    )

    def __repr__(self) -> str:
        return (
            f"<Signature id={self.id} user_id={self.user_id} "
            f"label={self.label!r} faiss_id={self.faiss_index_id}>"
        )


class MatchLog(TimestampMixin, Base):
    """
    Immutable audit record of every signature verification attempt.

    MatchLog rows are never soft-deleted; they form the compliance audit trail.
    They record the query image, the best-matching reference, the raw score,
    and the final pass/fail verdict.

    Attributes:
        id              : Auto-increment primary key.
        user_id         : FK to the user whose signatures were matched against.
        query_file_path : Path to the temporary query signature image.
        best_match_id   : FK to the Signature row that scored highest.
        score           : Cosine similarity score in range [0.0, 1.0].
        threshold_used  : The threshold value active at verification time.
        verdict         : True = MATCH (score >= threshold), False = NO MATCH.
        source_type     : 'image' or 'video'.
        ip_address      : Client IP for audit purposes.
        processing_ms   : Total pipeline processing time in milliseconds.
        error_message   : Populated if the pipeline raised an exception.
    """

    __tablename__ = "match_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_match_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("signatures.id", ondelete="SET NULL"), nullable=True
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), default="image", nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(50), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="match_logs")
    best_match_signature: Mapped["Signature | None"] = relationship(
        "Signature",
        back_populates="match_logs_as_best",
        foreign_keys=[best_match_id],
    )

    def __repr__(self) -> str:
        return (
            f"<MatchLog id={self.id} user_id={self.user_id} "
            f"verdict={self.verdict} score={self.score:.4f if self.score else None}>"
        )
