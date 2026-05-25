"""
app/db/repositories/video_repo.py
==================================
Repository for the `videos` collection.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.exceptions import DatabaseOperationError
from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository
from app.models.video import VideoDocument

logger = get_logger(__name__)


class VideoRepository(BaseRepository):
    collection_name = "videos"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    # ------------------------------------------------------------------ #
    # Domain-specific queries                                              #
    # ------------------------------------------------------------------ #

    async def get_by_video_id(self, video_id: str) -> Optional[dict]:
        """Fetch a video document by YouTube video ID."""
        return await self.find_one({"video_id": video_id})

    async def create_video(self, video: VideoDocument) -> str:
        """
        Insert a new video document.

        Returns:
            The MongoDB _id as a string.
        """
        return await self.insert_one(video.to_dict())

    async def video_exists(self, video_id: str) -> bool:
        """Check if we have already seen this video."""
        return await self.exists({"video_id": video_id})

    async def mark_scrape_completed(self, video_id: str, total_scraped: int) -> None:
        """Mark a video's comment scrape as fully complete."""
        await self.update_one(
            {"video_id": video_id},
            {
                "$set": {
                    "scrape_completed": True,
                    "comments_scraped": total_scraped,
                    "last_continuation_token": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def update_continuation_token(
        self, video_id: str, token: str, scraped_so_far: int
    ) -> None:
        """
        Persist the continuation token so a scrape can resume after
        interruption (crash, timeout, rate limit).
        """
        await self.update_one(
            {"video_id": video_id},
            {
                "$set": {
                    "last_continuation_token": token,
                    "comments_scraped": scraped_so_far,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def get_incomplete_scrapes(self) -> list[dict]:
        """
        Return videos that started scraping but never completed.
        Used by a recovery worker to re-queue interrupted jobs.
        """
        return await self.find_many(
            {
                "scrape_completed": False,
                "last_continuation_token": {"$ne": None},
            },
            limit=50,
        )
