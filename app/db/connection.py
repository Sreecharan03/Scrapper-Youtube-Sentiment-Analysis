"""
app/db/connection.py
====================
Async MongoDB connection manager using Motor.

WHY THIS FILE EXISTS:
  - Motor's AsyncIOMotorClient maintains a connection POOL, not a single
    socket. Creating one per request would exhaust Atlas connection limits
    within seconds under any real load.
  - This module manages a single, shared client instance with a clean
    connect/disconnect lifecycle tied to FastAPI's app lifespan.
  - All repositories import `get_database()` from here — they never create
    their own clients.

MOTOR vs PYMONGO:
  - pymongo (sync): what was in main.py. Blocks the event loop.
  - motor (async): wraps pymongo, yields control back to the event loop
    during I/O. Required for FastAPI/uvicorn async workers.

LIFECYCLE:
  connect_db()     → called in FastAPI lifespan startup
  get_database()   → called by repositories for every operation
  disconnect_db()  → called in FastAPI lifespan shutdown (clean pool drain)
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings
from app.core.exceptions import DatabaseConnectionError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Module-level state (private — use the functions below)              #
# ------------------------------------------------------------------ #
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_db() -> None:
    """
    Create the Motor client and verify connectivity via a ping.

    Called once at application startup (FastAPI lifespan).
    Raises DatabaseConnectionError if Atlas is unreachable — the app
    will refuse to start rather than serving requests with no database.
    """
    global _client, _database

    settings = get_settings()

    # Safety guard: don't create duplicate clients
    if _client is not None:
        logger.warning("db_already_connected", db=settings.mongodb_db_name)
        return

    logger.info("db_connecting", db=settings.mongodb_db_name, env=settings.app_env)

    try:
        _client = AsyncIOMotorClient(
            settings.mongodb_uri,
            # Timeout for initial connection attempt (milliseconds)
            serverSelectionTimeoutMS=10_000,
            # Max time to wait for a connection from the pool
            connectTimeoutMS=10_000,
            # Socket-level timeout for reads/writes
            socketTimeoutMS=30_000,
            # TLS configuration
            tls=True,
            tlsAllowInvalidCertificates=settings.mongodb_tls_allow_invalid_certs,
        )

        # Verify the connection is actually alive before declaring success.
        # Motor is lazy — AsyncIOMotorClient() doesn't connect until first use.
        await _client.admin.command("ping")

        _database = _client[settings.mongodb_db_name]

        logger.info(
            "db_connected",
            db=settings.mongodb_db_name,
            tls_strict=not settings.mongodb_tls_allow_invalid_certs,
        )

    except Exception as exc:
        # Clean up partially-initialised client
        if _client is not None:
            _client.close()
            _client = None
            _database = None

        raise DatabaseConnectionError(
            f"Failed to connect to MongoDB Atlas: {exc}",
            detail=f"URI host: {settings.mongodb_uri.split('@')[-1].split('/')[0]}",
        ) from exc


async def disconnect_db() -> None:
    """
    Close the Motor client and drain the connection pool.

    Called once at application shutdown (FastAPI lifespan).
    Safe to call even if connect_db() was never called.
    """
    global _client, _database

    if _client is None:
        logger.debug("db_disconnect_skipped", reason="no_active_client")
        return

    logger.info("db_disconnecting")
    _client.close()
    _client = None
    _database = None
    logger.info("db_disconnected")


def get_database() -> AsyncIOMotorDatabase:
    """
    Return the active database handle.

    Called by every repository method. Raises immediately if connect_db()
    was never called — prevents silent failures where code runs with no DB.

    Returns:
        AsyncIOMotorDatabase: The yt_scraper database handle.

    Raises:
        DatabaseConnectionError: If the database is not yet initialised.
    """
    if _database is None:
        raise DatabaseConnectionError(
            "Database is not initialised. "
            "Ensure connect_db() was called during application startup."
        )
    return _database


def get_client() -> AsyncIOMotorClient:
    """
    Return the raw Motor client (needed for transactions).

    Most code should use get_database(). Use get_client() only when you
    need to start a ClientSession for multi-document transactions.
    """
    if _client is None:
        raise DatabaseConnectionError(
            "MongoDB client is not initialised. "
            "Ensure connect_db() was called during application startup."
        )
    return _client
