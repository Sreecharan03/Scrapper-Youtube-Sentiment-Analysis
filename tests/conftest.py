"""
tests/conftest.py
==================
Shared pytest fixtures for the entire test suite.

FIXTURE SCOPES:
  - session:   Created once per pytest run (expensive resources like DB clients)
  - function:  Created/torn down per test (default — safest isolation)

ENVIRONMENT ISOLATION:
  - Tests use a separate test database ("yt_scraper_test") to avoid touching
    real data.
  - The `override_settings` fixture patches env vars for config tests.
  - `get_settings.cache_clear()` is called before tests that change env vars
    so the lru_cache doesn't return stale config.
"""

import asyncio
import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# ---- Patch env vars BEFORE any app imports that read config ----
os.environ.setdefault("MONGODB_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
os.environ.setdefault("MONGODB_DB_NAME", "yt_scraper_test")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


# ------------------------------------------------------------------ #
# Event loop                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def event_loop():
    """
    Provide a single event loop for the entire test session.
    Required for session-scoped async fixtures.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ------------------------------------------------------------------ #
# Database fixtures                                                    #
# ------------------------------------------------------------------ #

@pytest_asyncio.fixture(scope="session")
async def mongo_client() -> AsyncGenerator[AsyncIOMotorClient, None]:
    """
    Session-scoped Motor client pointing at the TEST database.
    Shared across all integration tests — created once, closed at the end.
    """
    from app.core.config import get_settings
    settings = get_settings()

    client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5_000,
        tls=True,
        tlsAllowInvalidCertificates=settings.mongodb_tls_allow_invalid_certs,
    )
    yield client
    client.close()


@pytest_asyncio.fixture(scope="session")
async def test_database(mongo_client: AsyncIOMotorClient) -> AsyncIOMotorDatabase:
    """
    Session-scoped handle to the yt_scraper_test database.
    Integration tests use this instead of get_database() to avoid
    touching real data.
    """
    from app.db.init_db import init_db
    db = mongo_client["yt_scraper_test"]
    await init_db(db)
    return db


@pytest_asyncio.fixture
async def clean_database(test_database: AsyncIOMotorDatabase):
    """
    Function-scoped fixture that yields the test DB and drops all
    test collections after each test.
    Use this for integration tests that write data.
    """
    yield test_database
    # Teardown: wipe test data after each test
    for collection_name in ["videos", "comments", "jobs"]:
        await test_database[collection_name].delete_many({})


# ------------------------------------------------------------------ #
# FastAPI test client                                                  #
# ------------------------------------------------------------------ #

@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client for FastAPI endpoint tests.
    Uses ASGI transport — no real server needed.
    """
    from main import app  # Import here to avoid circular deps at collection time
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ------------------------------------------------------------------ #
# Mock fixtures                                                        #
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_job_repo():
    """Mock JobRepository for unit-testing endpoints without a real DB."""
    repo = AsyncMock()
    repo.has_active_job.return_value = False
    repo.create_job.return_value = "507f1f77bcf86cd799439011"
    repo.get_job.return_value = {
        "_id": "507f1f77bcf86cd799439011",
        "video_id": "dQw4w9WgXcQ",
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "status": "pending",
        "comments_collected": 0,
        "created_at": __import__("datetime").datetime.utcnow(),
    }
    return repo
