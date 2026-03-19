"""
Pydantic Request & Response Schemas
======================================
Module  : schemas/signature.py
Purpose : Define strongly-typed data contracts for all API endpoints.
          Separates domain models (SQLAlchemy ORM) from API surface (Pydantic).

Author  : Signature Verifier Team
Version : 1.0.0
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─── User Schemas ─────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """Request body for POST /api/users/register."""
    name: str = Field(..., min_length=2, max_length=120, examples=["John Doe"])
    email: EmailStr = Field(..., examples=["john@example.com"])
    password: str = Field(..., min_length=8, max_length=128, examples=["SecurePass123!"])

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be blank or whitespace only.")
        return v.strip()


class UserResponse(BaseModel):
    """Response body for user creation and retrieval endpoints."""
    id: int
    name: str
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Response body for POST /api/auth/login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")


# ─── Signature Registration Schemas ──────────────────────────────────────────

class SignatureRegisterResponse(BaseModel):
    """Response body after successfully registering a reference signature."""
    signature_id: int
    user_id: int
    label: Optional[str]
    file_path: str
    faiss_id: Optional[int]
    created_at: datetime
    message: str = "Signature registered successfully."

    model_config = {"from_attributes": True}


class SignatureListItem(BaseModel):
    """One item in the list returned by GET /api/signatures/{user_id}."""
    id: int
    user_id: int
    label: Optional[str]
    file_path: str
    faiss_id: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}


class SignatureListResponse(BaseModel):
    """Response body for GET /api/signatures/{user_id}."""
    user_id: int
    total: int
    signatures: List[SignatureListItem]


# ─── Verification Schemas ─────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """
    Query parameters for POST /api/signatures/verify.
    (File is passed as multipart form data; these are the form fields.)
    """
    user_id: int = Field(..., gt=0, description="ID of the user to verify against.")
    label: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional label for the stored query image.",
    )


class ScoreBreakdown(BaseModel):
    """Per-reference similarity scores returned in the verification response."""
    signature_id: int
    score: float = Field(..., ge=-1.0, le=1.0)


class VerifyResponse(BaseModel):
    """Full response body from POST /api/signatures/verify."""
    user_id: int
    verdict: bool = Field(..., description="True = MATCH, False = NO MATCH")
    verdict_label: str = Field(..., description="Human-readable: 'MATCH' or 'NO MATCH'")
    score: float = Field(..., ge=-1.0, le=1.0, description="Best cosine similarity score")
    confidence: str = Field(..., description="Very High | High | Medium | Low | Very Low")
    threshold_used: float
    best_match_signature_id: Optional[int]
    source: str = Field(..., description="'image' or 'video'")
    match_strategy: str = Field(..., description="highest | lowest | average")
    score_breakdown: List[ScoreBreakdown]
    match_log_id: int
    processed_at: datetime

    @classmethod
    def from_match_result(
        cls,
        match_result,
        user_id: int,
        source: str,
        match_log_id: int,
    ) -> "VerifyResponse":
        """
        Factory method to build a VerifyResponse from a MatchResult.

        Args:
            match_result  : MatchResult instance from SignatureMatcher.
            user_id       : The user who was verified.
            source        : "image" or "video".
            match_log_id  : PK of the created MatchLog audit record.

        Returns:
            VerifyResponse
        """
        return cls(
            user_id=user_id,
            verdict=match_result.verdict,
            verdict_label="MATCH" if match_result.verdict else "NO MATCH",
            score=round(match_result.score, 6),
            confidence=match_result.confidence,
            threshold_used=match_result.threshold_used,
            best_match_signature_id=match_result.best_sig_id,
            source=source,
            match_strategy=match_result.match_strategy,
            score_breakdown=[
                ScoreBreakdown(signature_id=sig_id, score=round(score, 6))
                for sig_id, score in match_result.all_scores.items()
            ],
            match_log_id=match_log_id,
            processed_at=datetime.utcnow(),
        )


# ─── Match History Schemas ────────────────────────────────────────────────────

class MatchLogItem(BaseModel):
    """One item in the match history list."""
    id: int
    user_id: int
    best_match_id: Optional[int]
    query_path: str
    score: float
    threshold_used: float
    verdict: bool
    verdict_label: str
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_with_label(cls, log_orm) -> "MatchLogItem":
        """Build from ORM model, adding the computed verdict_label."""
        return cls(
            id=log_orm.id,
            user_id=log_orm.user_id,
            best_match_id=log_orm.best_match_id,
            query_path=log_orm.query_path,
            score=log_orm.score,
            threshold_used=log_orm.threshold_used,
            verdict=log_orm.verdict,
            verdict_label="MATCH" if log_orm.verdict else "NO MATCH",
            source=log_orm.source,
            created_at=log_orm.created_at,
        )


class MatchHistoryResponse(BaseModel):
    """Paginated match history response."""
    user_id: int
    total_returned: int
    limit: int
    offset: int
    logs: List[MatchLogItem]


# ─── Health Schema ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    version: str
    database: str
    is_model_loaded: bool
    uptime_seconds: float
    timestamp: datetime
