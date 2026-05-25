"""
app/db/repositories/scrape_batch_repo.py
==========================================
Repository for the scrape_batches collection.
"""

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository
from app.models.scrape_batch import BatchStatus, ScrapeBatchDocument

logger = get_logger(__name__)


class ScrapeBatchRepository(BaseRepository):
    collection_name = "scrape_batches"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    # ── Write ──────────────────────────────────────────────────────────────

    async def create_batch(self, batch: ScrapeBatchDocument) -> str:
        """Insert a new batch document. Returns MongoDB _id string."""
        return await self.insert_one(batch.to_dict())

    async def mark_running(self, batch_id: str, celery_task_id: str, worker: str) -> None:
        await self.update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {
                "status":          BatchStatus.RUNNING,
                "celery_task_id":  celery_task_id,
                "worker_hostname": worker,
                "started_at":      datetime.now(timezone.utc),
            }},
        )

    async def checkpoint(
        self,
        batch_id:       str,
        *,
        sub_batches_done:   int,
        comments_written:   int,
        duplicates_skipped: int,
        reply_tokens_found: int,
        current_token:      str,
    ) -> None:
        """
        Save progress after every 100-comment sub-batch.
        Called frequently — uses $set only on changed fields.
        """
        await self.update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {
                "sub_batches_done":   sub_batches_done,
                "comments_written":   comments_written,
                "duplicates_skipped": duplicates_skipped,
                "reply_tokens_found": reply_tokens_found,
                "token_at_end":       current_token,
            }},
        )

    async def mark_completed(self, batch_id: str, token_at_end: str, stats: dict) -> None:
        await self.update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {
                "status":            BatchStatus.COMPLETED,
                "token_at_end":      token_at_end,
                "completed_at":      datetime.now(timezone.utc),
                "comments_written":  stats.get("comments_written", 0),
                "duplicates_skipped":stats.get("duplicates_skipped", 0),
                "reply_tokens_found":stats.get("reply_tokens_found", 0),
                "sub_batches_done":  stats.get("sub_batches_done", 0),
            }},
        )

    async def mark_failed(self, batch_id: str, error: str) -> None:
        await self.update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {
                "status":       BatchStatus.FAILED,
                "completed_at": datetime.now(timezone.utc),
            },
             "$push": {"errors": {"error": error, "at": datetime.now(timezone.utc)}}},
        )

    async def mark_paused(self, batch_id: str, reason: str) -> None:
        await self.update_one(
            {"_id": ObjectId(batch_id)},
            {"$set": {"status": BatchStatus.PAUSED},
             "$push": {"errors": {"reason": reason, "at": datetime.now(timezone.utc)}}},
        )

    async def append_error(self, batch_id: str, error: str) -> None:
        """Record a transient error that was recovered — for audit trail."""
        await self.update_one(
            {"_id": ObjectId(batch_id)},
            {"$push": {"errors": {"error": error, "at": datetime.now(timezone.utc)}}},
        )

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_batch(self, batch_id: str) -> Optional[dict]:
        return await self.find_by_id(batch_id)

    async def get_batch_by_number(self, job_id: str, batch_number: int) -> Optional[dict]:
        return await self.find_one({"job_id": job_id, "batch_number": batch_number})

    async def list_batches_for_job(self, job_id: str) -> list[dict]:
        return await self.find_many(
            {"job_id": job_id},
            limit=500,
            sort=[("batch_number", 1)],
        )

    async def get_failed_batches(self, job_id: str) -> list[dict]:
        return await self.find_many(
            {"job_id": job_id, "status": {"$in": [BatchStatus.FAILED, BatchStatus.PAUSED]}},
            limit=50,
        )

    async def count_completed(self, job_id: str) -> int:
        return await self.count({"job_id": job_id, "status": BatchStatus.COMPLETED})
