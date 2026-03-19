"""
Authentication Service
========================
Module  : services/auth.py
Purpose : Password hashing, JWT creation and verification.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.config import get_settings
from backend.core.exceptions import AuthenticationError
from backend.core.logger import get_logger

log = get_logger("auth")
settings = get_settings()

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Password hashing and JWT token management."""

    @staticmethod
    def hash_password(plain: str) -> str:
        """Return a bcrypt hash of the plain-text password."""
        return _pwd_context.hash(plain)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        """Return True if plain matches the bcrypt hash."""
        return _pwd_context.verify(plain, hashed)

    @staticmethod
    def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
        """
        Create a signed JWT access token.

        Args:
            subject        : Typically the user's email or ID (str).
            expires_minutes: Override the default expiry from settings.

        Returns:
            str: Signed JWT string.
        """
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=expires_minutes or settings.TOKEN_EXPIRE_MINUTES
        )
        payload = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        log.debug(f"Access token created | sub={subject} | exp={expire.isoformat()}")
        return token

    @staticmethod
    def decode_token(token: str) -> str:
        """
        Decode and validate a JWT, returning the subject claim.

        Raises:
            AuthenticationError: If the token is invalid, expired, or malformed.
        """
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            sub: Optional[str] = payload.get("sub")
            if sub is None:
                raise AuthenticationError("Token missing subject claim.")
            return sub
        except JWTError as exc:
            log.warning(f"JWT decode failed | error={exc}")
            raise AuthenticationError(f"Invalid token: {exc}") from exc
