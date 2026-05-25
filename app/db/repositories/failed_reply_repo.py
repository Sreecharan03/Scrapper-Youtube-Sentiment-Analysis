"""
app/db/repositories/failed_reply_repo.py
==========================================
Repository for failed_replies collection.
"""

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.repositories.base import BaseRepository
from app.models.failed_reply import FailedReplyDocument, FailedReplyStatus


class FailedReplyRepository(BaseRepository):
    collection_name = "failed_replies"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    async def record_failure(self, doc: FailedReplyDocument) -> str:
        """
        Insert or update a failed-reply record.
        Uses upsert so repeated failures for the same comment increment attempts.
        """
        from bson import ObjectId
        result = await self._collection.find_one_and_update(
            {"job_id": doc.job_id, "comment_id": doc.comment_id},
            {
                "$set": {
                    "reply_token":      doc.reply_token,
                    "last_error":       doc.last_error,
                    "last_error_type":  doc.last_error_type,
                    "last_attempt_at":  datetime.now(timezone.utc),
                    "video_id":         doc.video_id,
                },
                "$inc": {"attempts": 1},
                "$setOnInsert": {
                    "status":     FailedReplyStatus.PENDING_RETRY,
                    "created_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
            return_document=True,
        )
        return str(result["_id"]) if result else ""

    async def mark_exhausted(self, job_id: str, comment_id: str) -> None:
        await self.update_one(
            {"job_id": job_id, "comment_id": comment_id},
            {"$set": {"status": FailedReplyStatus.EXHAUSTED}},
        )

    async def get_pending_retries(self, limit: int = 100) -> list[dict]:
        return await self.find_many(
            {"status": FailedReplyStatus.PENDING_RETRY},
            limit=limit,
            sort=[("created_at", 1)],
        )

    async def count_for_job(self, job_id: str) -> int:
        return await self.count({"job_id": job_id})
