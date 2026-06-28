"""
app/db/repositories/recommendation_repo.py
============================================
CRUD for the `recommendations` collection (Phase 3D).

Schema per document:
{
  video_id:               str (unique)
  status:                 "processing" | "completed" | "failed"
  generated_at:           datetime
  executive_summary:      str
  audience_stage:         str
  audience_mood:          str
  top_video_ideas:        [...]
  purchase_intent_signals: [...]
  content_series:         [...]
  risk_alerts:            [...]
  content_gaps:           [...]
  misconceptions:         [...]
  controversy_hotspots:   [...]
  unanswered_questions:   [...]
  error:                  str | null
}
"""

from datetime import datetime, timezone
from typing import Optional

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository

logger = get_logger(__name__)

RECOMMENDATIONS_COLLECTION = "recommendations"


class RecommendationRepository(BaseRepository):
    collection_name = RECOMMENDATIONS_COLLECTION

    async def get(self, video_id: str) -> Optional[dict]:
        return await self._collection.find_one(
            {"video_id": video_id}, {"_id": 0}
        )

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
        logger.info("recommendations_processing", video_id=video_id)

    async def mark_completed(self, video_id: str, data: dict) -> None:
        now = datetime.now(tz=timezone.utc)
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {"status": "completed", "generated_at": now, "error": None, **data}},
            upsert=True,
        )
        logger.info("recommendations_completed", video_id=video_id)

    async def mark_failed(self, video_id: str, error: str) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {"status": "failed", "error": error}},
            upsert=True,
        )
        logger.warning("recommendations_failed", video_id=video_id, error=error)
