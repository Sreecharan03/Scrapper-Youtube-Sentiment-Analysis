"""
app/db/repositories/classification_repo.py
============================================
Repository for the `comment_analysis` collection.

One document per video_id. Stores:
  - Classification job status (pending/processing/completed/failed)
  - Aggregate counts: sentiment breakdown, intent breakdown
  - Metadata: timestamps, version, error message

This is the document the dashboard reads to render charts.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository

logger = get_logger(__name__)


class ClassificationStatus:
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class ClassificationRepository(BaseRepository):
    collection_name = "comment_analysis"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    async def get_analysis(self, video_id: str) -> Optional[dict]:
        """Return the comment_analysis doc for a video, or None."""
        return await self.find_one({"video_id": video_id})

    async def get_status(self, video_id: str) -> Optional[str]:
        """Return just the status field, or None if doc doesn't exist."""
        doc = await self._collection.find_one({"video_id": video_id}, {"status": 1})
        return doc.get("status") if doc else None

    async def mark_processing(self, video_id: str, total_comments: int) -> None:
        """Upsert — creates doc if missing, sets status to processing."""
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                "video_id":       video_id,
                "status":         ClassificationStatus.PROCESSING,
                "total_comments": total_comments,
                "started_at":     datetime.now(timezone.utc),
                "error":          None,
            }},
            upsert=True,
        )
        logger.info("classification_marked_processing", video_id=video_id, total=total_comments)

    async def mark_completed(self, video_id: str, aggregates: dict) -> None:
        """Store computed aggregates and mark as completed."""
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                **aggregates,
                "status":       ClassificationStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "error":        None,
            }},
            upsert=True,
        )
        logger.info(
            "classification_marked_completed",
            video_id=video_id,
            classified=aggregates.get("classified_count"),
            failed=aggregates.get("failed_count"),
            skipped=aggregates.get("skipped_count"),
        )

    async def mark_failed(self, video_id: str, error: str) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                "status": ClassificationStatus.FAILED,
                "error":  error,
            }},
            upsert=True,
        )
        logger.error("classification_marked_failed", video_id=video_id, error=error)
