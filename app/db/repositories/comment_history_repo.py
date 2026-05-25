"""
app/db/repositories/comment_history_repo.py
=============================================
Repository for the comment_history collection (append-only).
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING

from app.db.repositories.base import BaseRepository
from app.models.comment_history import CommentHistoryDocument


class CommentHistoryRepository(BaseRepository):
    collection_name = "comment_history"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    async def archive_version(self, history: CommentHistoryDocument) -> str:
        """
        Archive the OLD version of a comment before updating it.
        The unique index on (comment_id, version) prevents double-archiving.
        """
        try:
            return await self.insert_one(history.to_dict())
        except Exception:
            # Duplicate key = already archived — not an error
            return ""

    async def get_history_for_comment(self, comment_id: str) -> list[dict]:
        """Return all archived versions for a comment, oldest first."""
        return await self.find_many(
            {"comment_id": comment_id},
            limit=100,
            sort=[("version", 1)],
        )

    async def get_recent_edits(self, video_id: str, limit: int = 50) -> list[dict]:
        """Return the most recently detected edits for a video."""
        return await self.find_many(
            {"video_id": video_id},
            limit=limit,
            sort=[("detected_at", DESCENDING)],
        )

    async def count_edits_for_video(self, video_id: str) -> int:
        return await self.count({"video_id": video_id})
