"""
Custom Exception Hierarchy
============================
Module  : core/exceptions.py
Purpose : Define domain-specific exceptions for clean, structured error handling
          across the entire application. All exceptions inherit from a single
          base class so callers can catch broadly or narrowly.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from typing import Optional


# ─── Base ─────────────────────────────────────────────────────────────────────

class SignatureVerifierError(Exception):
    """
    Root exception for the Signature Verifier application.

    All custom exceptions inherit from this class, enabling callers
    to catch any application error with a single except clause.

    Attributes:
        message (str)         : Human-readable description of the error.
        detail  (str | None)  : Optional technical detail for logging.
        status_code (int)     : HTTP status code hint for API layer.
    """

    def __init__(
        self,
        message: str,
        detail: Optional[str] = None,
        status_code: int = 500,
    ) -> None:
        self.message = message
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code})"
        )


# ─── Image / File Errors ──────────────────────────────────────────────────────

class ImageLoadError(SignatureVerifierError):
    """Raised when an image file cannot be read or decoded."""
    def __init__(self, path: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Failed to load image: {path}",
            detail=detail,
            status_code=422,
        )


class InvalidImageFormatError(SignatureVerifierError):
    """Raised when the uploaded file is not a supported image format."""
    def __init__(self, filename: str):
        super().__init__(
            message=f"Unsupported image format: {filename}. Use PNG, JPG, JPEG, or BMP.",
            status_code=415,
        )


class ImagePreprocessingError(SignatureVerifierError):
    """Raised when the OpenCV preprocessing pipeline fails."""
    def __init__(self, step: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Preprocessing failed at step: {step}",
            detail=detail,
            status_code=422,
        )


# ─── Video Errors ─────────────────────────────────────────────────────────────

class VideoLoadError(SignatureVerifierError):
    """Raised when a video file cannot be opened."""
    def __init__(self, path: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Failed to load video: {path}",
            detail=detail,
            status_code=422,
        )


class NoUsableFrameError(SignatureVerifierError):
    """Raised when no sufficiently sharp frame is found in a video."""
    def __init__(self):
        super().__init__(
            message="No usable frame found in video. Ensure the video shows a clear signature.",
            status_code=422,
        )


# ─── ML / Model Errors ────────────────────────────────────────────────────────

class ModelNotLoadedError(SignatureVerifierError):
    """Raised when inference is attempted before the model is initialized."""
    def __init__(self):
        super().__init__(
            message="ML model is not loaded. Check MODEL_WEIGHTS_PATH in .env.",
            status_code=503,
        )


class EmbeddingExtractionError(SignatureVerifierError):
    """Raised when the neural network fails to produce an embedding."""
    def __init__(self, detail: Optional[str] = None):
        super().__init__(
            message="Failed to extract embedding vector from signature image.",
            detail=detail,
            status_code=500,
        )


# ─── Matching / FAISS Errors ──────────────────────────────────────────────────

class VectorStoreError(SignatureVerifierError):
    """Raised when FAISS index operations fail."""
    def __init__(self, operation: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Vector store operation failed: {operation}",
            detail=detail,
            status_code=500,
        )


class NoReferenceSignatureError(SignatureVerifierError):
    """Raised when a user has no stored reference signatures to match against."""
    def __init__(self, user_id: int):
        super().__init__(
            message=f"No reference signatures found for user_id={user_id}. Register at least one signature first.",
            status_code=404,
        )


# ─── Database Errors ──────────────────────────────────────────────────────────

class DatabaseError(SignatureVerifierError):
    """Raised when a database operation fails unexpectedly."""
    def __init__(self, operation: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Database operation failed: {operation}",
            detail=detail,
            status_code=500,
        )


class RecordNotFoundError(SignatureVerifierError):
    """Raised when a requested DB record does not exist."""
    def __init__(self, entity: str, entity_id):
        super().__init__(
            message=f"{entity} with id={entity_id} not found.",
            status_code=404,
        )


# ─── Auth / User Errors ───────────────────────────────────────────────────────

class UserAlreadyExistsError(SignatureVerifierError):
    """Raised when attempting to register a duplicate email."""
    def __init__(self, email: str):
        super().__init__(
            message=f"User with email '{email}' already exists.",
            status_code=409,
        )


class AuthenticationError(SignatureVerifierError):
    """Raised when JWT validation fails."""
    def __init__(self, detail: str = "Invalid or expired token."):
        super().__init__(message=detail, status_code=401)
