"""
app/db/repositories/intent_summary_repo.py
============================================
CRUD for the `intent_summaries` collection.

One doc per video_id:
{
  video_id:              str (unique)
  status:                "processing" | "completed" | "failed"
  generated_at:          datetime
  comment_count_at_gen:  int   ← used to detect stale cache
  overall_summary:       str
  intent_summaries:      {intent: {summary, count}}
  error:                 str | null
}
"""

from datetime import datetime, timezone
from typing import Optional

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository

logger = get_logger(__name__)

INTENT_SUMMARIES_COLLECTION = "intent_summaries"


class IntentSummaryRepository(BaseRepository):
    collection_name = INTENT_SUMMARIES_COLLECTION

    async def get(self, video_id: str) -> Optional[dict]:
        return await self._collection.find_one({"video_id": video_id}, {"_id": 0})

    async def get_status(self, video_id: str) -> Optional[str]:
        doc = await self._collection.find_one(
            {"video_id": video_id}, {"status": 1, "_id": 0}
        )
        return doc["status"] if doc else None

    async def mark_processing(self, video_id: str) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                "status":       "processing",
                "started_at":   datetime.now(tz=timezone.utc),
                "generated_at": None,
                "error":        None,
            }},
            upsert=True,
        )

    async def mark_completed(self, video_id: str, comment_count: int, data: dict) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                "status":               "completed",
                "generated_at":         datetime.now(tz=timezone.utc),
                "comment_count_at_gen": comment_count,
                "error":                None,
                "overall_summary":      data.get("overall_summary", ""),
                "intent_summaries":     data.get("intent_summaries", {}),
            }},
            upsert=True,
        )
        logger.info("intent_summaries_completed", video_id=video_id)

    async def mark_failed(self, video_id: str, error: str) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {"status": "failed", "error": error}},
            upsert=True,
        )
        logger.warning("intent_summaries_failed", video_id=video_id, error=error)
