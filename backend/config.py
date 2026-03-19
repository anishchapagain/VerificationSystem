"""
Application Configuration
===========================
Module  : config.py
Purpose : Load and validate all environment variables using Pydantic Settings.
          Single source of truth for configuration across the application.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    All fields have sensible defaults for local development. In production,
    override via environment variables or a .env file.

    Attributes:
        APP_NAME        : Display name of the application.
        APP_VERSION     : Semantic version string.
        DEBUG           : Enable FastAPI debug mode and verbose logging.
        LOG_LEVEL       : Loguru log level (DEBUG | INFO | WARNING | ERROR).
        DATABASE_URL    : PostgreSQL connection DSN.
        MODEL_WEIGHTS   : Path to the trained Siamese Network checkpoint.
        EMBEDDING_DIM   : Dimension of the embedding vector produced by the model.
        MATCH_THRESHOLD : Cosine similarity threshold above which = MATCH.
        STORAGE_PATH    : Directory for persisted signature image files.
        FAISS_INDEX     : Path to the FAISS index file on disk.
        ALLOWED_ORIGINS : CORS origins for the Streamlit frontend.
        SECRET_KEY      : JWT signing secret (keep this secure in production!).
        TOKEN_EXPIRE_MIN: JWT token lifetime in minutes.
        VIDEO_STRIDE    : Process every Nth frame from a video input.
        VIDEO_TOP_FRAMES: Number of sharpest frames used for ensemble voting.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Signature Verifier API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/signature_db",
        description="Async PostgreSQL connection string (asyncpg driver required).",
    )

    # ── ML Model ─────────────────────────────────────────────────────────────
    MODEL_WEIGHTS_PATH: str = Field(
        default="weights/siamese_best.pt",
        description="Path to the trained Siamese Network .pt checkpoint file.",
    )
    EMBEDDING_DIM: int = Field(default=512, ge=64, le=2048)
    MATCH_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)

    # ── Storage ───────────────────────────────────────────────────────────────
    SIGNATURE_STORAGE_PATH: str = "storage/signatures"
    FAISS_INDEX_PATH: str = "storage/embeddings/index.faiss"

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",
    ]

    # ── Auth ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(
        default="changeme-use-a-long-random-string-in-production",
        description="HMAC secret for JWT signing. Change in production!",
    )
    ALGORITHM: str = "HS256"
    TOKEN_EXPIRE_MINUTES: int = 60

    # ── Video ─────────────────────────────────────────────────────────────────
    VIDEO_FRAME_STRIDE: int = Field(default=5, ge=1)
    VIDEO_TOP_FRAMES: int = Field(default=3, ge=1, le=10)

    # ── Image Preprocessing ───────────────────────────────────────────────────
    IMAGE_TARGET_WIDTH: int = 256
    IMAGE_TARGET_HEIGHT: int = 128
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    MAX_FILE_SIZE_MB: int = 10

    @field_validator("SIGNATURE_STORAGE_PATH", "FAISS_INDEX_PATH", mode="before")
    @classmethod
    def ensure_parent_dirs(cls, v: str) -> str:
        """Auto-create storage directories if they do not exist."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == "":  # It's a directory path
            path.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Uses lru_cache so the .env file is only parsed once per process lifetime.
    Inject this into FastAPI endpoints via Depends(get_settings).

    Returns:
        Settings: Validated application configuration object.

    Example:
        >>> settings = get_settings()
        >>> print(settings.APP_NAME)
    """
    return Settings()
