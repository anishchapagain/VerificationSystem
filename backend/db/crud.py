"""
CRUD — Database Read/Write Operations
========================================
Module  : db/crud.py
Purpose : All database interactions abstracted into a CRUDManager class.
          Routes never touch SQLAlchemy directly — they call CRUD methods.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import DatabaseError, RecordNotFoundError, UserAlreadyExistsError
from backend.core.logger import get_logger
from backend.db.models import MatchLog, Signature, User

log = get_logger("crud")


class UserCRUD:
    """
    Data access layer for the User table.

    All methods accept an AsyncSession injected by FastAPI's Depends(get_db).
    Raises domain-specific exceptions instead of raw SQLAlchemy errors.
    """

    @staticmethod
    async def create(
        db: AsyncSession,
        name: str,
        email: str,
        hashed_password: str,
    ) -> User:
        """
        Insert a new user record.

        Args:
            db              : Active async database session.
            name            : Full display name.
            email           : Unique email address.
            hashed_password : Bcrypt-hashed password string.

        Returns:
            User: The newly created and refreshed ORM instance.

        Raises:
            UserAlreadyExistsError : If the email is already registered.
            DatabaseError          : On any other DB failure.
        """
        try:
            existing = await UserCRUD.get_by_email(db, email)
            if existing:
                raise UserAlreadyExistsError(email)

            user = User(name=name, email=email, hashed_password=hashed_password)
            db.add(user)
            await db.flush()
            await db.refresh(user)
            log.info(f"User created | id={user.id} email={email}")
            return user
        except UserAlreadyExistsError:
            raise
        except Exception as exc:
            log.error(f"UserCRUD.create failed | email={email} | error={exc}")
            raise DatabaseError("create user", detail=str(exc)) from exc

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """Fetch a user by primary key. Returns None if not found."""
        try:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as exc:
            raise DatabaseError("get_user_by_id", detail=str(exc)) from exc

    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Fetch an active user by email. Returns None if not found."""
        try:
            result = await db.execute(
                select(User).where(User.email == email, User.is_active == True)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            raise DatabaseError("get_user_by_email", detail=str(exc)) from exc

    @staticmethod
    async def deactivate(db: AsyncSession, user_id: int) -> None:
        """Soft-delete a user by setting is_active = False."""
        try:
            await db.execute(
                update(User).where(User.id == user_id).values(is_active=False)
            )
            log.info(f"User deactivated | id={user_id}")
        except Exception as exc:
            raise DatabaseError("deactivate_user", detail=str(exc)) from exc


class SignatureCRUD:
    """Data access layer for the Signature table."""

    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: int,
        file_path: str,
        embedding: np.ndarray,
        faiss_id: Optional[int] = None,
        label: Optional[str] = None,
    ) -> Signature:
        """
        Persist a new reference signature.

        Args:
            db        : Active async session.
            user_id   : Owner's primary key.
            file_path : Relative path to the stored image file.
            embedding : 512-D float32 numpy array.
            faiss_id  : Row index in the FAISS flat index.
            label     : Optional human label (e.g. "Primary").

        Returns:
            Signature: The persisted ORM instance.
        """
        try:
            sig = Signature(
                user_id=user_id,
                file_path=file_path,
                embedding=embedding.astype(np.float32).tobytes(),
                faiss_id=faiss_id,
                label=label,
            )
            db.add(sig)
            await db.flush()
            await db.refresh(sig)
            log.info(f"Signature stored | id={sig.id} user_id={user_id} faiss_id={faiss_id}")
            return sig
        except Exception as exc:
            log.error(f"SignatureCRUD.create failed | user_id={user_id} | error={exc}")
            raise DatabaseError("create_signature", detail=str(exc)) from exc

    @staticmethod
    async def get_by_user(db: AsyncSession, user_id: int) -> List[Signature]:
        """Return all active signatures for a given user, newest first."""
        try:
            result = await db.execute(
                select(Signature)
                .where(Signature.user_id == user_id, Signature.is_active == True)
                .order_by(Signature.created_at.desc())
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseError("get_signatures_by_user", detail=str(exc)) from exc

    @staticmethod
    async def get_by_id(db: AsyncSession, sig_id: int) -> Optional[Signature]:
        """Fetch a single signature by primary key."""
        try:
            result = await db.execute(select(Signature).where(Signature.id == sig_id))
            return result.scalar_one_or_none()
        except Exception as exc:
            raise DatabaseError("get_signature_by_id", detail=str(exc)) from exc

    @staticmethod
    async def get_embeddings_by_user(
        db: AsyncSession, user_id: int
    ) -> List[tuple[int, np.ndarray]]:
        """
        Return (signature_id, embedding_array) tuples for all active user signatures.

        Used by the matcher to load vectors for cosine similarity comparison.
        """
        try:
            sigs = await SignatureCRUD.get_by_user(db, user_id)
            return [
                (s.id, np.frombuffer(s.embedding, dtype=np.float32))
                for s in sigs
            ]
        except Exception as exc:
            raise DatabaseError("get_embeddings_by_user", detail=str(exc)) from exc

    @staticmethod
    async def soft_delete(db: AsyncSession, sig_id: int) -> None:
        """Soft-delete a signature (sets is_active = False)."""
        try:
            await db.execute(
                update(Signature).where(Signature.id == sig_id).values(is_active=False)
            )
            log.info(f"Signature soft-deleted | id={sig_id}")
        except Exception as exc:
            raise DatabaseError("soft_delete_signature", detail=str(exc)) from exc

    @staticmethod
    async def update_faiss_id(db: AsyncSession, sig_id: int, faiss_id: int) -> None:
        """Update the FAISS row index after rebuilding the index."""
        try:
            await db.execute(
                update(Signature).where(Signature.id == sig_id).values(faiss_id=faiss_id)
            )
        except Exception as exc:
            raise DatabaseError("update_faiss_id", detail=str(exc)) from exc


class MatchLogCRUD:
    """Data access layer for the MatchLog table."""

    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: int,
        query_path: str,
        score: float,
        threshold_used: float,
        verdict: bool,
        source: str = "image",
        best_match_id: Optional[int] = None,
    ) -> MatchLog:
        """
        Append a new verification event to the immutable audit log.

        Args:
            user_id        : User being verified.
            query_path     : Stored path of the uploaded query image.
            score          : Cosine similarity score [0.0 – 1.0].
            threshold_used : The threshold applied at verification time.
            verdict        : True = MATCH, False = NO MATCH.
            source         : "image" or "video".
            best_match_id  : PK of the closest Signature, if any.

        Returns:
            MatchLog: The created ORM instance.
        """
        try:
            log_entry = MatchLog(
                user_id=user_id,
                query_path=query_path,
                score=score,
                threshold_used=threshold_used,
                verdict=verdict,
                source=source,
                best_match_id=best_match_id,
            )
            db.add(log_entry)
            await db.flush()
            await db.refresh(log_entry)
            verdict_str = "MATCH" if verdict else "NO MATCH"
            log.info(
                f"MatchLog created | id={log_entry.id} user_id={user_id} "
                f"score={score:.4f} verdict={verdict_str}"
            )
            return log_entry
        except Exception as exc:
            raise DatabaseError("create_match_log", detail=str(exc)) from exc

    @staticmethod
    async def get_by_user(
        db: AsyncSession,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MatchLog]:
        """Return paginated match logs for a user, newest first."""
        try:
            result = await db.execute(
                select(MatchLog)
                .where(MatchLog.user_id == user_id)
                .order_by(MatchLog.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseError("get_match_logs_by_user", detail=str(exc)) from exc
