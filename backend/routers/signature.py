"""
Signature Router — Core API Endpoints
========================================
Module  : routers/signature.py
Purpose : FastAPI router handling signature registration, verification,
          listing, and deletion. All business logic is delegated to services.

Author  : Signature Verifier Team
Version : 1.0.0
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.core.exceptions import (
    ImageLoadError,
    InvalidImageFormatError,
    ModelNotLoadedError,
    NoReferenceSignatureError,
    RecordNotFoundError,
    SignatureVerifierError,
)
from backend.core.logger import get_logger
from backend.db.crud import MatchLogCRUD, SignatureCRUD, UserCRUD
from backend.db.database import get_db
from backend.models.siamese_net import ModelManager
from backend.schemas.signature import (
    MatchHistoryResponse,
    MatchLogItem,
    SignatureListResponse,
    SignatureRegisterResponse,
    VerifyResponse,
)
from backend.services.matcher import SignatureMatcher
from backend.services.preprocessor import SignaturePreprocessor
from backend.services.video_handler import VideoSignatureExtractor

log = get_logger("router.signature")

router = APIRouter(prefix="/api/signatures", tags=["Signatures"])

# ─── Model Singleton ──────────────────────────────────────────────────────────
# Loaded ONCE when the module is first imported.
# All requests share the same model instance.
# This ensures register and verify always use identical weights,
# so stored embeddings and query embeddings live in the same vector space.

_model_manager: ModelManager = None


def get_model_manager(settings: Settings = Depends(get_settings)) -> ModelManager:
    """
    Return the shared ModelManager singleton.

    The model is loaded once on first request and reused for all subsequent
    requests. This guarantees that:
      - /register and /verify always use the same model
      - Stored embeddings and query embeddings are always comparable
      - No per-request model loading overhead
    """
    global _model_manager
    if _model_manager is None or not _model_manager.is_loaded:
        log.info("Initialising ModelManager singleton...")
        manager = ModelManager(
            weights_path=settings.MODEL_WEIGHTS_PATH,
            embedding_dim=settings.EMBEDDING_DIM,
        )
        manager.load()
        _model_manager = manager
        log.info(
            f"ModelManager singleton ready | "
            f"device={manager.device} | dim={manager.embedding_dim}"
        )
    return _model_manager


def get_preprocessor(settings: Settings = Depends(get_settings)) -> SignaturePreprocessor:
    """Provide a configured SignaturePreprocessor."""
    return SignaturePreprocessor(
        target_width=settings.IMAGE_TARGET_WIDTH,
        target_height=settings.IMAGE_TARGET_HEIGHT,
    )


def get_matcher(settings: Settings = Depends(get_settings)) -> SignatureMatcher:
    """Provide a configured SignatureMatcher with the current threshold."""
    return SignatureMatcher(threshold=settings.MATCH_THRESHOLD)


# ─── Helper ───────────────────────────────────────────────────────────────────

async def _save_upload(
    upload: UploadFile,
    storage_path: str,
    allowed_extensions: List[str],
    max_size_mb: int,
) -> str:
    """
    Save an uploaded file to disk after validating extension and size.

    Args:
        upload            : FastAPI UploadFile instance.
        storage_path      : Directory path for persistent storage.
        allowed_extensions: List of permitted file extensions (e.g. ['.png']).
        max_size_mb       : Maximum file size in megabytes.

    Returns:
        str: Absolute path to the saved file.

    Raises:
        HTTPException 415 : Unsupported file type.
        HTTPException 413 : File too large.
    """
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    content = await upload.read()
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {max_size_mb} MB limit.",
        )

    dest_dir = Path(storage_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{uuid.uuid4().hex}{suffix}"

    with open(dest_path, "wb") as f:
        f.write(content)

    log.debug(f"File saved | path={dest_path} | size={len(content)} bytes")
    return str(dest_path)


def _handle_domain_error(exc: SignatureVerifierError) -> HTTPException:
    """Convert a domain exception to the appropriate HTTPException."""
    return HTTPException(status_code=exc.status_code, detail=exc.message)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=SignatureRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a reference signature",
    description=(
        "Upload a handwritten signature image to store as a reference for a user. "
        "The image is preprocessed, embedded, and stored in the database + FAISS index."
    ),
)
async def register_signature(
    file: UploadFile = File(..., description="Signature image (PNG/JPG/BMP)."),
    user_id: int = Form(..., gt=0, description="ID of the user this signature belongs to."),
    label: Optional[str] = Form(None, max_length=100),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    preprocessor: SignaturePreprocessor = Depends(get_preprocessor),
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Register a new reference signature for a user.

    Steps:
    1. Validate and persist the uploaded file.
    2. Run the preprocessing pipeline (grayscale → binary → crop → resize).
    3. Extract a 512-D embedding vector via the Siamese encoder.
    4. Store the embedding and metadata in PostgreSQL.
    5. Return the created signature record.
    """
    log.info(f"Register signature | user_id={user_id} | file={file.filename}")

    try:
        # 1. Save upload to disk
        file_path = await _save_upload(
            upload=file,
            storage_path=settings.SIGNATURE_STORAGE_PATH,
            allowed_extensions=settings.ALLOWED_IMAGE_EXTENSIONS,
            max_size_mb=settings.MAX_FILE_SIZE_MB,
        )

        # 2. Preprocess
        prep_result = preprocessor.run(file_path)

        # 3. Extract embedding
        embedding = model_manager.extract_embedding(prep_result.image)

        # 4. Persist to database
        sig = await SignatureCRUD.create(
            db=db,
            user_id=user_id,
            file_path=file_path,
            embedding=embedding,
            label=label,
        )

        log.info(f"Signature registered | sig_id={sig.id} | user_id={user_id}")
        return SignatureRegisterResponse(
            signature_id=sig.id,
            user_id=sig.user_id,
            label=sig.label,
            file_path=sig.file_path,
            faiss_id=sig.faiss_id,
            created_at=sig.created_at,
        )

    except HTTPException:
        raise
    except (InvalidImageFormatError, ImageLoadError) as exc:
        raise _handle_domain_error(exc)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except SignatureVerifierError as exc:
        raise _handle_domain_error(exc)
    except Exception as exc:
        log.exception(f"Unexpected error in register_signature | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a signature against stored references",
    description=(
        "Upload an image or video containing a handwritten signature. "
        "The system computes cosine similarity against the user's stored references "
        "and returns a MATCH / NO MATCH verdict with a confidence score."
    ),
)
async def verify_signature(
    file: UploadFile = File(..., description="Signature image or video file."),
    user_id: int = Form(..., gt=0),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    preprocessor: SignaturePreprocessor = Depends(get_preprocessor),
    model_manager: ModelManager = Depends(get_model_manager),
    matcher: SignatureMatcher = Depends(get_matcher),
):
    """
    Verify a query signature against a user's stored reference signatures.

    Steps:
    1. Detect input modality (image vs. video).
    2. Save the file and run the preprocessing / frame extraction pipeline.
    3. Extract embedding(s).
    4. Load all reference embeddings for the user from the database.
    5. Run cosine similarity matching (ensemble if video).
    6. Log the result to the audit MatchLog table.
    7. Return the full VerifyResponse.
    """
    log.info(f"Verify request | user_id={user_id} | file={file.filename}")

    suffix = Path(file.filename).suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov", ".mkv"}
    source = "video" if is_video else "image"

    allowed_ext = settings.ALLOWED_IMAGE_EXTENSIONS + [".mp4", ".avi", ".mov", ".mkv"]

    try:
        # 1. Save upload
        file_path = await _save_upload(
            upload=file,
            storage_path=settings.SIGNATURE_STORAGE_PATH + "/queries",
            allowed_extensions=allowed_ext,
            max_size_mb=settings.MAX_FILE_SIZE_MB,
        )

        # 2–3. Preprocess and extract embedding(s)
        if is_video:
            extractor = VideoSignatureExtractor(preprocessor=preprocessor)
            prep_results = extractor.extract(file_path)
            embeddings = [model_manager.extract_embedding(r.image) for r in prep_results]
        else:
            prep_result = preprocessor.run(file_path)
            embeddings = [model_manager.extract_embedding(prep_result.image)]

        # 4. Load reference embeddings from DB
        references = await SignatureCRUD.get_embeddings_by_user(db, user_id)

        # 5. Match
        if is_video and len(embeddings) > 1:
            match_result = matcher.ensemble_match(embeddings, references, user_id)
        else:
            match_result = matcher.match(embeddings[0], references, user_id)

        # 6. Write audit log
        log_entry = await MatchLogCRUD.create(
            db=db,
            user_id=user_id,
            query_path=file_path,
            score=match_result.score,
            threshold_used=match_result.threshold_used,
            verdict=match_result.verdict,
            source=source,
            best_match_id=match_result.best_sig_id,
        )

        log.info(
            f"Verification complete | user_id={user_id} | "
            f"verdict={'MATCH' if match_result.verdict else 'NO MATCH'} | "
            f"score={match_result.score:.4f}"
        )

        # 7. Build and return response
        return VerifyResponse.from_match_result(
            match_result=match_result,
            user_id=user_id,
            source=source,
            match_log_id=log_entry.id,
        )

    except HTTPException:
        raise
    except NoReferenceSignatureError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except SignatureVerifierError as exc:
        raise _handle_domain_error(exc)
    except Exception as exc:
        log.exception(f"Unexpected error in verify_signature | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get(
    "/{user_id}",
    response_model=SignatureListResponse,
    summary="List all reference signatures for a user",
)
async def list_signatures(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return all active reference signatures stored for a given user."""
    try:
        sigs = await SignatureCRUD.get_by_user(db, user_id)
        return SignatureListResponse(
            user_id=user_id,
            total=len(sigs),
            signatures=sigs,
        )
    except SignatureVerifierError as exc:
        raise _handle_domain_error(exc)
    except Exception as exc:
        log.exception(f"list_signatures failed | user_id={user_id} | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.delete(
    "/{signature_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a reference signature",
)
async def delete_signature(
    signature_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a stored reference signature by ID."""
    try:
        sig = await SignatureCRUD.get_by_id(db, signature_id)
        if not sig:
            raise HTTPException(
                status_code=404,
                detail=f"Signature {signature_id} not found.",
            )
        await SignatureCRUD.soft_delete(db, signature_id)
        log.info(f"Signature deleted | id={signature_id}")
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(f"delete_signature failed | id={signature_id} | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get(
    "/history/{user_id}",
    response_model=MatchHistoryResponse,
    summary="Get verification history for a user",
)
async def match_history(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Return paginated match verification history for a given user."""
    try:
        logs = await MatchLogCRUD.get_by_user(db, user_id, limit=limit, offset=offset)
        return MatchHistoryResponse(
            user_id=user_id,
            total_returned=len(logs),
            limit=limit,
            offset=offset,
            logs=[MatchLogItem.from_orm_with_label(l) for l in logs],
        )
    except Exception as exc:
        log.exception(f"match_history failed | user_id={user_id} | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")