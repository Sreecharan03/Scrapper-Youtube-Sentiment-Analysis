"""
app/db/repositories/comment_repo.py
=====================================
Repository for the `comments` collection.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository
from app.models.comment import CommentDocument

logger = get_logger(__name__)


class CommentRepository(BaseRepository):
    collection_name = "comments"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    # ------------------------------------------------------------------ #
    # Domain-specific queries                                              #
    # ------------------------------------------------------------------ #

    async def bulk_insert_comments(
        self, comments: list[CommentDocument]
    ) -> tuple[int, int]:
        """
        Insert a batch of comments, ignoring duplicates (by comment_id + video_id).

        The `comments` index is a unique compound index on (video_id, comment_id),
        so re-inserting the same comment on a re-scrape is silently skipped
        via `ordered=False` in insert_many.

        Returns:
            (inserted_count, duplicate_count): counts for monitoring/logging.
        """
        if not comments:
            return 0, 0

        docs = [c.to_dict() for c in comments]
        try:
            result = await self._collection.insert_many(docs, ordered=False)
            inserted = len(result.inserted_ids)
            duplicates = len(docs) - inserted
            logger.info(
                "comments_batch_inserted",
                inserted=inserted,
                duplicates=duplicates,
            )
            return inserted, duplicates
        except Exception as exc:
            # BulkWriteError contains partial results — extract what succeeded
            if hasattr(exc, "details"):
                inserted = exc.details.get("nInserted", 0)
                duplicates = len(docs) - inserted
                logger.warning(
                    "comments_batch_partial",
                    inserted=inserted,
                    duplicates=duplicates,
                    error=str(exc),
                )
                return inserted, duplicates
            raise

    async def get_comments_for_video(
        self,
        video_id: str,
        *,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "published_at",
        descending: bool = True,
    ) -> list[dict]:
        """
        Paginated retrieval of comments for a video.

        Args:
            sort_by: Field to sort by ("published_at", "like_count", "scraped_at")
            descending: True = newest/most-liked first
        """
        direction = DESCENDING if descending else ASCENDING
        return await self.find_many(
            {"video_id": video_id},
            skip=skip,
            limit=limit,
            sort=[(sort_by, direction)],
        )

    async def count_comments_for_video(self, video_id: str) -> int:
        """Count how many comments we have stored for a video."""
        return await self.count({"video_id": video_id})

    async def get_top_comments(
        self, video_id: str, *, limit: int = 10
    ) -> list[dict]:
        """Fetch the most-liked top-level comments for a video."""
        return await self.find_many(
            {"video_id": video_id, "is_reply": False},
            limit=limit,
            sort=[("like_count", DESCENDING)],
        )

    async def comment_exists(self, comment_id: str, video_id: str) -> bool:
        """Check for a specific comment — used to avoid duplicate scraping."""
        return await self.exists({"comment_id": comment_id, "video_id": video_id})
