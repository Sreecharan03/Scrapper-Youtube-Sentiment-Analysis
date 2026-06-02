"""
app/db/repositories/summary_repo.py
=====================================
MongoDB repository for the `summaries` collection.
One summary per video_id — re-generate overwrites.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger
from app.models.summary import SummaryStatus

logger = get_logger(__name__)
COLLECTION = "summaries"


class SummaryRepository:

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    async def mark_generating(self, video_id: str) -> None:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {"status": SummaryStatus.GENERATING, "updated_at": now, "error": None}},
            upsert=True,
        )

    async def mark_completed(self, video_id: str, data: dict) -> None:
        """
        Write all LLM output fields and transition to COMPLETED.
        `data` is the dict returned by SummaryService.generate().
        """
        now = datetime.now(timezone.utc)
        meta = data.pop("_meta", {})

        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {
                **data,
                "model":               meta.get("model"),
                "critique_severity":   meta.get("critique_severity"),
                "critique_notes":      meta.get("critique_notes"),
                "total_input_tokens":  meta.get("total_input_tokens", 0),
                "total_output_tokens": meta.get("total_output_tokens", 0),
                "status":              SummaryStatus.COMPLETED,
                "error":               None,
                "generated_at":        now,
                "updated_at":          now,
            }},
            upsert=True,
        )

    async def mark_failed(self, video_id: str, error: str) -> None:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {"status": SummaryStatus.FAILED, "error": error, "updated_at": now}},
            upsert=True,
        )

    async def get_summary(self, video_id: str) -> Optional[dict]:
        return await self._col.find_one({"video_id": video_id})

    async def get_status(self, video_id: str) -> Optional[str]:
        doc = await self._col.find_one({"video_id": video_id}, {"status": 1})
        return doc["status"] if doc else None
