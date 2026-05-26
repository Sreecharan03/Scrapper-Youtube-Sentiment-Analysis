"""
app/db/repositories/transcript_repo.py
=======================================
MongoDB repository for the `transcripts` collection.

DESIGN:
  - One document per video_id (upsert on re-fetch).
  - Status transitions are explicit methods — callers never do raw $set.
  - get_transcript() returns the raw dict (like other repos) — the API layer
    formats it for callers.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger
from app.models.transcript import TranscriptDocument, TranscriptStatus

logger = get_logger(__name__)

COLLECTION = "transcripts"


class TranscriptRepository:

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    # ── Create / upsert ───────────────────────────────────────────────────

    async def create_or_replace(self, doc: TranscriptDocument) -> None:
        """
        Insert or fully replace an existing transcript for video_id.
        Used on every fetch (re-fetch overwrites old data).
        """
        await self._col.replace_one(
            {"video_id": doc.video_id},
            doc.to_dict(),
            upsert=True,
        )
        logger.debug("transcript_upserted", video_id=doc.video_id, status=doc.status)

    # ── Status transitions ────────────────────────────────────────────────

    async def mark_fetching(self, video_id: str) -> None:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {
                "status":     TranscriptStatus.FETCHING,
                "updated_at": now,
            }},
            upsert=True,
        )

    async def mark_completed(self, video_id: str, data: dict) -> None:
        """
        Write all fetched data and transition to COMPLETED.

        `data` must contain:
          original_language_code, original_language_name, is_auto_generated,
          available_languages, original_segments, english_segments,
          is_translated, segment_count, total_duration_secs
        """
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {
                **data,
                "status":     TranscriptStatus.COMPLETED,
                "error":      None,
                "fetched_at": now,
                "updated_at": now,
            }},
            upsert=True,
        )

    async def mark_unavailable(self, video_id: str, reason: str) -> None:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {
                "status":     TranscriptStatus.UNAVAILABLE,
                "error":      reason,
                "updated_at": now,
            }},
            upsert=True,
        )

    async def mark_failed(self, video_id: str, error: str) -> None:
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"video_id": video_id},
            {"$set": {
                "status":     TranscriptStatus.FAILED,
                "error":      error,
                "updated_at": now,
            }},
            upsert=True,
        )

    # ── Reads ─────────────────────────────────────────────────────────────

    async def get_transcript(self, video_id: str) -> Optional[dict]:
        """Return the transcript document for a video, or None."""
        return await self._col.find_one({"video_id": video_id})

    async def get_status(self, video_id: str) -> Optional[str]:
        """Return just the status string, or None if no document exists."""
        doc = await self._col.find_one(
            {"video_id": video_id},
            {"status": 1},
        )
        return doc["status"] if doc else None
