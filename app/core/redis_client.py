"""
app/core/redis_client.py
=========================
Async Redis connection manager using redis.asyncio.

WHY THIS FILE EXISTS:
  - redis.asyncio.Redis is the async client built into redis-py >= 4.2.
    It does NOT block the event loop — safe to use inside FastAPI handlers.
  - One connection pool is shared across the entire app (same pattern as
    Motor for MongoDB). Creating a new client per request is wasteful and
    will exhaust Redis Cloud's connection limit.
  - This module owns the entire lifecycle: connect → use → disconnect.
    Nothing else creates Redis clients directly.

SYNC vs ASYNC:
  - The user's original test code used `redis.Redis` (synchronous).
  - We use `redis.asyncio.Redis` which is identical in API but every
    command is awaitable, keeping the event loop free during network I/O.
  - Celery uses its OWN sync connection to Redis (separate from this client).
    This client is for the FastAPI application layer only.

DB SEPARATION:
  - This client connects to DB index `redis_cache_db` (default: 0).
  - Celery broker (DB 1) and results (DB 2) are managed by Celery internally.
  - Keeping them separate means a FLUSHDB on the cache never wipes Celery state.

LIFECYCLE:
  connect_redis()     → called in FastAPI lifespan startup
  get_redis()         → called by CacheManager and any code needing direct access
  disconnect_redis()  → called in FastAPI lifespan shutdown
"""

from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.exceptions import AppBaseError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Custom exception for Redis errors                                    #
# ------------------------------------------------------------------ #

class RedisConnectionError(AppBaseError):
    """Cannot establish or maintain connection to Redis."""
    pass


# ------------------------------------------------------------------ #
# Module-level state (private — access via functions below)           #
# ------------------------------------------------------------------ #

_redis_client: Optional[aioredis.Redis] = None


async def connect_redis() -> None:
    """
    Create and verify the async Redis client.

    Called once at application startup (FastAPI lifespan).
    Uses a connection pool internally — redis.asyncio.Redis manages the
    pool automatically based on `max_connections`.

    Raises:
        RedisConnectionError: If Redis is unreachable at startup.
    """
    global _redis_client

    if _redis_client is not None:
        logger.warning("redis_already_connected")
        return

    settings = get_settings()
    logger.info(
        "redis_connecting",
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_cache_db,
        ssl=settings.redis_ssl,
    )

    try:
        _redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_cache_db,
            username=settings.redis_username,
            password=settings.redis_password or None,   # Pass None not ""
            ssl=settings.redis_ssl,
            decode_responses=True,         # All values are str — consistent with user's code
            socket_connect_timeout=10,     # Connection attempt timeout (seconds)
            socket_timeout=10,             # Read/write timeout per command
            retry_on_timeout=True,         # Auto-retry on socket timeout
            max_connections=5,             # Pool size — kept low for managed Redis
            health_check_interval=30,      # Background ping every 30s to keep alive
        )

        # Eagerly verify the connection — don't wait for first command
        await _redis_client.ping()

        logger.info(
            "redis_connected",
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_cache_db,
        )

    except Exception as exc:
        # Clean up on failure
        if _redis_client is not None:
            await _redis_client.aclose()
            _redis_client = None

        raise RedisConnectionError(
            f"Failed to connect to Redis at {settings.redis_host}:{settings.redis_port}",
            detail=str(exc),
        ) from exc


async def disconnect_redis() -> None:
    """
    Close the Redis client and drain the connection pool.

    Called at application shutdown. Safe to call even if connect_redis()
    was never called (e.g., if startup failed mid-way).
    """
    global _redis_client

    if _redis_client is None:
        logger.debug("redis_disconnect_skipped", reason="no_active_client")
        return

    logger.info("redis_disconnecting")
    await _redis_client.aclose()
    _redis_client = None
    logger.info("redis_disconnected")


def get_redis() -> aioredis.Redis:
    """
    Return the active async Redis client.

    Called by CacheManager and any code that needs direct Redis access.
    Raises immediately if called before connect_redis() — prevents silent
    failures where cache operations silently do nothing.

    Returns:
        redis.asyncio.Redis: The active client (DB 0 / cache DB).

    Raises:
        RedisConnectionError: If connect_redis() was not called at startup.
    """
    if _redis_client is None:
        raise RedisConnectionError(
            "Redis client is not initialised. "
            "Ensure connect_redis() was called during application startup."
        )
    return _redis_client
