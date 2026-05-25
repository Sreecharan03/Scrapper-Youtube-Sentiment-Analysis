"""
tests/integration/test_db_connection.py
=========================================
Integration tests for MongoDB connectivity.

These tests hit a REAL database (yt_scraper_test).
They are separated from unit tests so CI can skip them when
Atlas is not reachable (e.g., PR checks that only run unit tests).

MARKER: integration — requires real MongoDB Atlas connection.

Run only integration tests:
  pytest -m integration

Run only unit tests (no network):
  pytest -m unit
"""

import pytest
import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.connection import connect_db, disconnect_db, get_database
from app.db.init_db import COMMENTS_COLLECTION, JOBS_COLLECTION, VIDEOS_COLLECTION, init_db


@pytest.mark.integration
class TestMongoDBConnection:

    @pytest_asyncio.fixture(autouse=True)
    async def setup_connection(self, test_database):
        """Use the session-scoped test database from conftest.py."""
        self.db = test_database

    async def test_database_is_reachable(self):
        """Atlas must respond to a ping."""
        result = await self.db.client.admin.command("ping")
        assert result.get("ok") == 1.0

    async def test_collections_exist_after_init(self):
        """All three collections must exist after init_db() runs."""
        await init_db(self.db)
        collections = await self.db.list_collection_names()
        assert VIDEOS_COLLECTION in collections
        assert COMMENTS_COLLECTION in collections
        assert JOBS_COLLECTION in collections

    async def test_indexes_created_on_videos(self):
        """The unique video_id index must exist on the videos collection."""
        await init_db(self.db)
        index_info = await self.db[VIDEOS_COLLECTION].index_information()
        index_names = list(index_info.keys())
        assert "idx_video_id_unique" in index_names

    async def test_indexes_created_on_comments(self):
        """The unique (video_id, comment_id) compound index must exist."""
        await init_db(self.db)
        index_info = await self.db[COMMENTS_COLLECTION].index_information()
        assert "idx_video_comment_unique" in index_info

    async def test_init_db_is_idempotent(self):
        """Calling init_db() twice must not raise or create duplicate indexes."""
        await init_db(self.db)
        await init_db(self.db)  # Should be a no-op
        # If we get here without exception, the test passes

    async def test_insert_and_find(self, clean_database):
        """Basic write/read cycle must work end-to-end."""
        from datetime import datetime, timezone
        doc = {
            "video_id": "test_abc123",
            "url": "https://youtube.com/watch?v=test_abc123",
            "scrape_completed": False,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        result = await clean_database[VIDEOS_COLLECTION].insert_one(doc)
        assert result.inserted_id is not None

        fetched = await clean_database[VIDEOS_COLLECTION].find_one(
            {"video_id": "test_abc123"}
        )
        assert fetched is not None
        assert fetched["video_id"] == "test_abc123"
