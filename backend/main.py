"""
FastAPI Application Entry Point
=================================
Module  : backend/main.py
Purpose : Bootstrap the FastAPI application — database auto-creation,
          table init, middleware, exception handlers, router registration.

Author  : Signature Verifier Team
Version : 2.0.0

Run directly (no Docker needed):
    uvicorn backend.main:app --reload --port 8000

Or via Python:
    python -m uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.core.exceptions import SignatureVerifierError
from backend.core.logger import get_logger, setup_logger
from backend.db.database import (
    close_db,
    ensure_database_exists,
    init_db,
)
from backend.routers import health, signature, users

settings = get_settings()

# Logger must be set up before anything that uses it
setup_logger(
    log_level   = settings.LOG_LEVEL,
    log_to_file = True,
    enable_json = False,
)
log = get_logger("main")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application startup and shutdown logic.

    Startup sequence:
        1. ensure_database_exists()
              Connects to PostgreSQL maintenance DB ('postgres').
              Creates signature_db if it does not exist.
              Logs clearly if PostgreSQL is not running.

        2. init_db()
              Creates all ORM tables inside signature_db.
              Idempotent — safe to run on every restart.

        3. Log startup summary.

    Shutdown:
        - Dispose SQLAlchemy connection pool.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  {settings.APP_NAME}  v{settings.APP_VERSION}")
    log.info(f"  DEBUG      : {settings.DEBUG}")
    log.info(f"  LOG LEVEL  : {settings.LOG_LEVEL}")
    log.info(f"  DATABASE   : {settings.DATABASE_URL.split('@')[-1]}")
    log.info(f"  MODEL      : {settings.MODEL_WEIGHTS_PATH}")
    log.info(f"  THRESHOLD  : {settings.MATCH_THRESHOLD}")
    log.info(f"  DEVICE     : {settings.DEVICE if hasattr(settings, 'DEVICE') else 'auto'}")
    log.info("=" * 60)

    # Step 1 — Auto-create the database if it does not exist
    try:
        await ensure_database_exists()
    except Exception as exc:
        log.critical(
            f"Cannot connect to PostgreSQL. "
            f"Is it installed and running? Error: {exc}"
        )
        raise   # Abort startup — no point continuing without a DB

    # Step 2 — Create tables inside the database
    try:
        await init_db()
    except Exception as exc:
        log.critical(f"Table creation failed | {exc}")
        raise

    log.info("Startup complete — ready to accept requests.")
    log.info(f"  Swagger UI  : http://localhost:8000/docs")
    log.info(f"  Health      : http://localhost:8000/health")
    log.info(f"  Frontend    : http://localhost:8501")

    yield  # ── App is running ──────────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Shutting down...")
    await close_db()
    log.info(f"{settings.APP_NAME} stopped cleanly.")


# ─── Application Factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""

    app = FastAPI(
        title       = settings.APP_NAME,
        version     = settings.APP_VERSION,
        description = (
            "REST API for handwritten signature registration and verification "
            "using a Siamese Neural Network and cosine similarity matching."
        ),
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        lifespan    = lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.ALLOWED_ORIGINS,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Domain Exception Handler ──────────────────────────────────────────────
    @app.exception_handler(SignatureVerifierError)
    async def domain_exception_handler(
        request: Request, exc: SignatureVerifierError
    ) -> JSONResponse:
        log.warning(
            f"Domain error | {request.url.path} | "
            f"{type(exc).__name__} | {exc.message}"
        )
        return JSONResponse(
            status_code = exc.status_code,
            content     = {"detail": exc.message, "error_type": type(exc).__name__},
        )

    # ── Generic Exception Handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.exception(f"Unexpected error | {request.url.path} | {exc}")
        return JSONResponse(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            content     = {"detail": "An unexpected internal error occurred."},
        )

    # ── Request Logging Middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        log.info(f"→ {request.method} {request.url.path}")
        response = await call_next(request)
        log.info(f"← {request.method} {request.url.path} | {response.status_code}")
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(users.router)
    app.include_router(signature.router)

    log.info("Routers registered: health, users, signatures")
    return app


app = create_app()


# ─── Direct Run ───────────────────────────────────────────────────────────────
# Allows: python backend/main.py
# (uvicorn backend.main:app --reload is still preferred for development)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        workers = 1,
    )