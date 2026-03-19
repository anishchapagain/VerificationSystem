"""
Users Router — Registration & Authentication
==============================================
Module  : routers/users.py
Purpose : User registration, login, and profile endpoints.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import UserAlreadyExistsError, SignatureVerifierError
from backend.core.logger import get_logger
from backend.db.crud import UserCRUD
from backend.db.database import get_db
from backend.schemas.signature import TokenResponse, UserCreate, UserResponse
from backend.services.auth import AuthService

log = get_logger("router.users")
router = APIRouter(prefix="/api/users", tags=["Users"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new user account.

    Validates uniqueness of email, hashes the password with bcrypt,
    and persists the user record to PostgreSQL.
    """
    try:
        hashed = AuthService.hash_password(payload.password)
        user = await UserCRUD.create(
            db=db,
            name=payload.name,
            email=payload.email,
            hashed_password=hashed,
        )
        log.info(f"User registered | id={user.id} | email={user.email}")
        return user
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=exc.message)
    except SignatureVerifierError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        log.exception(f"register_user failed | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Obtain a JWT access token",
)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate with email + password and receive a JWT bearer token.

    The token must be passed in the Authorization header for protected endpoints:
        Authorization: Bearer <token>
    """
    try:
        user = await UserCRUD.get_by_email(db, form.username)
        if not user or not AuthService.verify_password(form.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = AuthService.create_access_token(subject=str(user.id))
        log.info(f"User logged in | id={user.id}")
        from backend.config import get_settings
        settings = get_settings()
        return TokenResponse(
            access_token=token,
            expires_in=settings.TOKEN_EXPIRE_MINUTES * 60,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(f"login failed | error={exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get("/me/{user_id}", response_model=UserResponse, summary="Get user profile")
async def get_profile(user_id: int, db: AsyncSession = Depends(get_db)):
    """Return a user's profile information."""
    user = await UserCRUD.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return user
