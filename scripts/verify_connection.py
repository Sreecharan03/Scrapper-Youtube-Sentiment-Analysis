"""
scripts/verify_connection.py
==============================
Standalone connection verification script.

Migrated from: main.py (the original MongoDB ping test)

PURPOSE:
  - Run this manually to confirm MongoDB Atlas connectivity from Lightning AI.
  - This is a UTILITY SCRIPT, not part of the application.
  - It reads from .env (same as the app) but runs independently of FastAPI.

WHY NOT IN app/:
  - Scripts like this are one-off dev tools — they should never be imported
    by production code. Keeping them in scripts/ enforces this boundary.

USAGE:
  python scripts/verify_connection.py

WHAT IT TESTS:
  1. .env loads correctly and MONGODB_URI is set
  2. Motor async client can connect to Atlas
  3. Ping command succeeds
  4. DB and collections are accessible
  5. Index creation works (calls init_db)
"""

import asyncio
import sys
from pathlib import Path

# ---- Add project root to path so app/ imports work ----
# This is only needed for scripts/ — app code never does this.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.init_db import init_db

setup_logging()
logger = get_logger("verify_connection")


async def verify() -> bool:
    """
    Run all connectivity checks. Returns True if everything passes.
    """
    settings = get_settings()
    logger.info("verification_started", env=settings.app_env, db=settings.mongodb_db_name)

    client = None
    try:
        # ---- Step 1: Create client ----
        logger.info("step_1_creating_client")
        client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=10_000,
            tls=True,
            tlsAllowInvalidCertificates=settings.mongodb_tls_allow_invalid_certs,
        )

        # ---- Step 2: Ping ----
        logger.info("step_2_pinging_atlas")
        await client.admin.command("ping")
        logger.info("step_2_ping_success")

        # ---- Step 3: Access database ----
        logger.info("step_3_accessing_database", db=settings.mongodb_db_name)
        db = client[settings.mongodb_db_name]
        collections = await db.list_collection_names()
        logger.info("step_3_existing_collections", collections=collections)

        # ---- Step 4: Initialize indexes ----
        logger.info("step_4_initialising_indexes")
        await init_db(db)
        logger.info("step_4_indexes_ok")

        # ---- Step 5: Final collection list ----
        collections_after = await db.list_collection_names()
        logger.info(
            "verification_passed",
            collections=sorted(collections_after),
            tls_strict=not settings.mongodb_tls_allow_invalid_certs,
        )
        return True

    except Exception as exc:
        logger.error(
            "verification_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        return False

    finally:
        if client:
            client.close()
            logger.info("client_closed")


if __name__ == "__main__":
    success = asyncio.run(verify())
    print("\n" + ("✅ Connection verified — all checks passed." if success
                  else "❌ Connection failed — check logs above."))
    sys.exit(0 if success else 1)
