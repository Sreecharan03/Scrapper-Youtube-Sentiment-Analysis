"""
main.py
========
Application entry point — FastAPI app factory and lifespan manager.

This file should stay THIN:
  - Create the FastAPI app
  - Register the lifespan (startup/shutdown hooks)
  - Mount the API router
  - Register exception handlers
  - Nothing else

All business logic lives in app/.

STARTUP COMMAND:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

PRODUCTION STARTUP:
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.cache import cache
from app.core.config import get_settings
from app.core.exceptions import (
    AppBaseError,
    DatabaseConnectionError,
    JobAlreadyExistsError,
    JobNotFoundError,
    AppValidationError,
)
from app.core.logging import get_logger, setup_logging
from app.core.redis_client import connect_redis, disconnect_redis
from app.db.connection import connect_db, disconnect_db, get_database
from app.db.init_db import init_db

# ------------------------------------------------------------------ #
# Logging must be configured before any other app code runs           #
# ------------------------------------------------------------------ #
setup_logging()
logger = get_logger(__name__)

settings = get_settings()


# ------------------------------------------------------------------ #
# Lifespan: startup & shutdown hooks                                  #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before `yield` runs at startup; code after runs at shutdown.

    Startup order matters:
      1. Validate config (production guards)
      2. Connect to MongoDB Atlas
      3. Initialize collections and indexes
    """
    # ---- Startup ----
    logger.info(
        "app_starting",
        name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )

    # Production safety guard: TLS must be strict in production
    if settings.is_production and settings.mongodb_tls_allow_invalid_certs:
        raise RuntimeError(
            "SECURITY: mongodb_tls_allow_invalid_certs=true is not allowed in production. "
            "Fix your TLS configuration before deploying."
        )

    # ---- Connect MongoDB → Initialize indexes ----
    await connect_db()
    await init_db(get_database())

    # ---- Connect Redis ----
    # Order: MongoDB first (jobs/comments), Redis second (cache/sessions).
    # If Redis fails, startup fails — don't serve traffic without a cache layer.
    await connect_redis()

    logger.info(
        "app_ready",
        env=settings.app_env,
        redis_host=settings.redis_host,
        redis_db=settings.redis_cache_db,
    )

    yield  # ← Application runs here

    # ---- Shutdown (reverse order of startup) ----
    logger.info("app_shutting_down")
    await disconnect_redis()
    await disconnect_db()
    logger.info("app_stopped")


# ------------------------------------------------------------------ #
# FastAPI app factory                                                  #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="YT Scraper API",
    description=(
        "Production-grade YouTube comment extraction platform. "
        "Submit scrape jobs, track progress, and query collected comments."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    # Disable docs in production (optional — remove if you want them)
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)


# ------------------------------------------------------------------ #
# Routers                                                              #
# ------------------------------------------------------------------ #

app.include_router(api_router)


# ------------------------------------------------------------------ #
# Exception handlers                                                   #
# ------------------------------------------------------------------ #

@app.exception_handler(DatabaseConnectionError)
async def db_connection_error_handler(request: Request, exc: DatabaseConnectionError):
    logger.error("database_connection_error", path=request.url.path, error=exc.message)
    return JSONResponse(
        status_code=503,
        content={"error": "service_unavailable", "message": "Database is unavailable."},
    )


@app.exception_handler(JobNotFoundError)
async def job_not_found_handler(request: Request, exc: JobNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "message": exc.message},
    )


@app.exception_handler(JobAlreadyExistsError)
async def job_exists_handler(request: Request, exc: JobAlreadyExistsError):
    return JSONResponse(
        status_code=409,
        content={"error": "conflict", "message": exc.message},
    )


@app.exception_handler(AppValidationError)
async def validation_error_handler(request: Request, exc: AppValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "message": exc.message},
    )


@app.exception_handler(AppBaseError)
async def generic_app_error_handler(request: Request, exc: AppBaseError):
    logger.error("unhandled_app_error", error_type=type(exc).__name__, error=exc.message)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["System"])
async def health_check():
    """
    Lightweight liveness probe.
    Returns 200 immediately — does NOT check DB (use /health/ready for that).
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
    }


@app.get("/health/ready", tags=["System"])
async def readiness_check():
    """
    Readiness probe — verifies both MongoDB and Redis connectivity.
    Returns 503 if either is unreachable.
    Used by load balancers / orchestrators to decide if traffic should route here.
    """
    checks: dict[str, str] = {}
    all_ok = True

    # ---- MongoDB check ----
    try:
        db = get_database()
        await db.client.admin.command("ping")
        checks["mongodb"] = "connected"
    except Exception as exc:
        logger.error("readiness_mongodb_failed", error=str(exc))
        checks["mongodb"] = "unreachable"
        all_ok = False

    # ---- Redis check ----
    redis_ok = await cache.ping()
    checks["redis"] = "connected" if redis_ok else "unreachable"
    if not redis_ok:
        all_ok = False

    status_code = 200 if all_ok else 503
    body = {
        "status": "ready" if all_ok else "not_ready",
        **checks,
    }

    if not all_ok:
        return JSONResponse(status_code=status_code, content=body)
    return body
