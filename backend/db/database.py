"""
Async Database Engine & Session Factory
=========================================
Module  : db/database.py
Purpose : Configure SQLAlchemy async engine and session lifecycle for PostgreSQL.
          Automatically creates the target database if it does not exist.
          Provides the get_db dependency injected into all FastAPI route handlers.

Author  : Signature Verifier Team
Version : 2.0.0
"""

import re
from typing import AsyncGenerator

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings
from backend.core.exceptions import DatabaseError
from backend.core.logger import get_logger

log      = get_logger("database")
settings = get_settings()


# ─── ORM Base ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all ORM models.
    All model classes in db/models.py must inherit from this Base.
    """
    pass


# ─── Auto Database Creation ───────────────────────────────────────────────────

def _parse_dsn(url: str) -> dict:
    """
    Parse a postgresql+asyncpg DSN into its components.

    Handles:
        postgresql+asyncpg://user:pass@host:port/dbname
        postgresql+asyncpg://user:pass@host/dbname   (defaults to port 5432)

    Returns:
        dict with keys: user, password, host, port, database
    """
    clean   = url.replace("postgresql+asyncpg://", "postgresql://")
    pattern = (
        r"postgresql://"
        r"(?P<user>[^:]+):(?P<password>[^@]+)@"
        r"(?P<host>[^:/]+)(?::(?P<port>\d+))?/"
        r"(?P<database>.+)"
    )
    match = re.match(pattern, clean)
    if not match:
        raise ValueError(f"Cannot parse DATABASE_URL: {url}")

    return {
        "user":     match.group("user"),
        "password": match.group("password"),
        "host":     match.group("host"),
        "port":     int(match.group("port") or 5432),
        "database": match.group("database"),
    }


async def ensure_database_exists() -> None:
    """
    Check if the target database exists on the PostgreSQL server.
    If it does not exist, create it automatically.

    This runs at startup BEFORE the SQLAlchemy engine is used, so a
    missing database never causes a crash.

    How it works:
        1. Parses DATABASE_URL for connection details.
        2. Connects to the 'postgres' maintenance database (always exists).
        3. Queries pg_database for the target database name.
        4. Creates the database if it is not found.
        5. Closes the maintenance connection.

    The main SQLAlchemy engine then connects normally to the now-guaranteed
    target database.

    Raises:
        DatabaseError: If PostgreSQL is unreachable, credentials are wrong,
                       or database creation fails.
    """
    try:
        dsn     = _parse_dsn(settings.DATABASE_URL)
        db_name = dsn["database"]

        log.info(
            f"Checking database | name='{db_name}' | "
            f"host={dsn['host']}:{dsn['port']}"
        )

        # Connect to the maintenance database — this is always safe
        conn = await asyncpg.connect(
            host     = dsn["host"],
            port     = dsn["port"],
            user     = dsn["user"],
            password = dsn["password"],
            database = "postgres",       # maintenance DB — always exists
        )

        try:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                db_name,
            )

            if exists:
                log.info(f"Database '{db_name}' already exists.")
            else:
                # CREATE DATABASE must run outside a transaction block
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                log.info(f"Database '{db_name}' created successfully.")

        finally:
            await conn.close()

    except asyncpg.InvalidPasswordError:
        msg = (
            "PostgreSQL authentication failed. "
            "Check the username and password in DATABASE_URL inside your .env file."
        )
        log.critical(msg)
        raise DatabaseError("ensure_database_exists", detail=msg)

    except OSError as exc:
        dsn_tail = settings.DATABASE_URL.split("@")[-1]
        msg = (
            f"Cannot reach PostgreSQL at {dsn_tail}. "
            f"Make sure PostgreSQL is installed and running. Error: {exc}"
        )
        log.critical(msg)
        raise DatabaseError("ensure_database_exists", detail=msg)

    except DatabaseError:
        raise

    except Exception as exc:
        log.critical(f"ensure_database_exists unexpected error | {exc}")
        raise DatabaseError("ensure_database_exists", detail=str(exc)) from exc


# ─── Engine ───────────────────────────────────────────────────────────────────

def _build_engine() -> AsyncEngine:
    """
    Build the async SQLAlchemy engine.

    Always called after ensure_database_exists() so the target
    database is guaranteed to exist.
    """
    try:
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo         = settings.DEBUG,
            pool_size    = 10,
            max_overflow = 20,
            pool_timeout = 30,
            pool_recycle = 3600,
            pool_pre_ping= True,
        )
        log.info(
            f"DB engine ready | {settings.DATABASE_URL.split('@')[-1]}"
        )
        return engine
    except Exception as exc:
        log.critical(f"Engine creation failed | {exc}")
        raise DatabaseError("engine creation", detail=str(exc)) from exc


engine: AsyncEngine = _build_engine()

AsyncSessionLocal = async_sessionmaker(
    bind           = engine,
    class_         = AsyncSession,
    expire_on_commit = False,
    autocommit     = False,
    autoflush      = False,
)


# ─── Request-scoped Session Dependency ───────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields one DB session per HTTP request.
    Commits on success, rolls back on any exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            log.error(f"DB session rolled back | {exc}")
            raise DatabaseError("session", detail=str(exc)) from exc
        finally:
            await session.close()


# ─── Lifecycle Helpers ────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Create all ORM-defined tables (idempotent — safe on every startup).
    Called from the FastAPI lifespan, AFTER ensure_database_exists().
    """
    try:
        async with engine.begin() as conn:
            from backend.db import models  # noqa: F401
            await conn.run_sync(Base.metadata.create_all)
        log.info("All tables created / verified.")
    except Exception as exc:
        log.critical(f"init_db failed | {exc}")
        raise DatabaseError("init_db", detail=str(exc)) from exc


async def check_db_health() -> bool:
    """Run SELECT 1 — used by the /health endpoint."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.warning(f"DB health check failed | {exc}")
        return False


async def close_db() -> None:
    """Dispose the engine pool on shutdown."""
    await engine.dispose()
    log.info("DB engine disposed.")