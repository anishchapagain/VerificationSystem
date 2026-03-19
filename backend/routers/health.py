"""
Health Check Router
====================
Module  : routers/health.py
Purpose : Expose a /health endpoint for load balancers, uptime monitors,
          and container orchestration readiness/liveness probes.

Author  : Signature Verifier Team
Version : 1.0.0
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends
from pathlib import Path

from backend.config import Settings, get_settings
from backend.core.logger import get_logger
from backend.db.database import check_db_health
from backend.schemas.signature import HealthResponse

log = get_logger("router.health")
router = APIRouter(tags=["Health"])

_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description=(
        "Returns service status, database connectivity, model load state, "
        "and uptime. Suitable for liveness and readiness probes."
    ),
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Perform a full system health check.

    Checks:
    - Database reachability (SELECT 1)
    - Model weights file existence on disk
    - Application uptime

    Returns:
        HealthResponse with overall status 'healthy' or 'degraded'.
    """
    db_ok = await check_db_health()
    is_model_loaded = Path(settings.MODEL_WEIGHTS_PATH).exists()
    uptime = round(time.time() - _START_TIME, 2)

    overall_status = "healthy" if (db_ok and is_model_loaded) else "degraded"

    log.debug(
        f"Health check | status={overall_status} | db={db_ok} | "
        f"model={is_model_loaded} | uptime={uptime}s"
    )

    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        database="connected" if db_ok else "unreachable",
        is_model_loaded=is_model_loaded,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow(),
    )
