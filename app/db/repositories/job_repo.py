"""
app/db/repositories/job_repo.py
=================================
Repository for the jobs collection — full lifecycle management.
"""

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository
from app.models.job import JobDocument, JobStatus

logger = get_logger(__name__)


class JobRepository(BaseRepository):
    collection_name = "jobs"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    # ── Create / Read ──────────────────────────────────────────────────────

    async def create_job(self, job: JobDocument) -> str:
        return await self.insert_one(job.to_dict())

    async def get_job(self, job_id: str) -> Optional[dict]:
        return await self.find_by_id(job_id)

    async def get_job_for_video(self, video_id: str) -> Optional[dict]:
        results = await self.find_many(
            {"video_id": video_id},
            limit=1,
            sort=[("created_at", -1)],
        )
        return results[0] if results else None

    async def has_active_job(self, video_id: str) -> bool:
        return await self.exists({
            "video_id": video_id,
            "status": {"$in": list(JobStatus.ACTIVE_STATUSES)},
        })

    async def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> list[dict]:
        f = {"status": status} if status else {}
        return await self.find_many(f, skip=skip, limit=limit, sort=[("created_at", -1)])

    # ── Status transitions ─────────────────────────────────────────────────

    async def mark_fetching_meta(self, job_id: str, celery_task_id: str) -> None:
        await self._set(job_id, {
            "status":          JobStatus.FETCHING_META,
            "phase":           "fetching_video_metadata",
            "celery_task_id":  celery_task_id,
            "started_at":      datetime.now(timezone.utc),
        })

    async def mark_scraping_tlcs(
        self, job_id: str, total_comments_expected: Optional[int]
    ) -> None:
        await self._set(job_id, {
            "status": JobStatus.SCRAPING_TLCS,
            "phase":  "scraping_top_level_comments",
            "total_comments_expected": total_comments_expected,
        })

    async def mark_finalizing(self, job_id: str) -> None:
        await self._set(job_id, {
            "status": JobStatus.FINALIZING,
            "phase":  "finalizing",
        })

    async def mark_completed(self, job_id: str, total_scraped: int) -> None:
        await self._set(job_id, {
            "status":             JobStatus.COMPLETED,
            "phase":              "completed",
            "comments_collected": total_scraped,
            "completed_at":       datetime.now(timezone.utc),
        })

    async def mark_paused_batch_failed(
        self, job_id: str, batch_number: int, error: str
    ) -> None:
        await self._set(job_id, {
            "status":        JobStatus.PAUSED_BATCH_FAILED,
            "phase":         f"paused_batch_{batch_number}_failed",
            "error_message": error,
            "error_type":    "BatchFailed",
        })

    async def mark_paused_token_expired(self, job_id: str) -> None:
        await self._set(job_id, {
            "status":        JobStatus.PAUSED_TOKEN_EXPIRED,
            "phase":         "paused_token_expired",
            "error_message": "Continuation token expired. Restart from last batch.",
            "error_type":    "TokenExpired",
        })

    async def mark_failed_permanent(self, job_id: str, reason: str) -> None:
        await self._set(job_id, {
            "status":        JobStatus.FAILED_PERMANENT,
            "phase":         "failed",
            "error_message": reason,
            "error_type":    "PermanentFailure",
            "completed_at":  datetime.now(timezone.utc),
        })

    # ── Progress updates ───────────────────────────────────────────────────

    async def increment_comments_collected(self, job_id: str, count: int) -> None:
        await self.update_one(
            {"_id": ObjectId(job_id)},
            {"$inc": {"comments_collected": count},
             "$set": {"updated_at": datetime.now(timezone.utc)}},
        )

    async def set_batch_progress(
        self, job_id: str, current_batch: int, total_completed: int
    ) -> None:
        await self._set(job_id, {
            "current_batch_number":    current_batch,
            "total_batches_completed": total_completed,
        })

    async def record_tlc_completed(self, job_id: str) -> None:
        await self._set(job_id, {"tlc_completed": True})

    async def increment_reply_tokens_found(self, job_id: str, count: int) -> None:
        await self.update_one(
            {"_id": ObjectId(job_id)},
            {"$inc": {"reply_tokens_found": count}},
        )

    async def increment_reply_tokens_completed(self, job_id: str) -> None:
        await self.update_one(
            {"_id": ObjectId(job_id)},
            {"$inc": {"reply_tokens_completed": 1}},
        )

    async def increment_reply_tokens_failed(self, job_id: str) -> None:
        await self.update_one(
            {"_id": ObjectId(job_id)},
            {"$inc": {"reply_tokens_failed": 1}},
        )

    # ── Recovery helpers ───────────────────────────────────────────────────

    async def resume_job(self, job_id: str) -> None:
        """Operator-triggered resume from a PAUSED state."""
        await self._set(job_id, {
            "status":        JobStatus.SCRAPING_TLCS,
            "phase":         "resumed_scraping",
            "error_message": None,
            "error_type":    None,
        })
        await self.update_one(
            {"_id": ObjectId(job_id)},
            {"$inc": {"retry_count": 1}},
        )

    async def get_stalled_jobs(self, stall_threshold_minutes: int = 30) -> list[dict]:
        """Find RUNNING jobs that haven't made progress in N minutes."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=stall_threshold_minutes)
        return await self.find_many(
            {
                "status": {"$in": list(JobStatus.ACTIVE_STATUSES - {JobStatus.PENDING})},
                "updated_at": {"$lt": cutoff},
            },
            limit=50,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    async def _set(self, job_id: str, fields: dict) -> None:
        fields["updated_at"] = datetime.now(timezone.utc)
        await self.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": fields},
        )

    @staticmethod
    def _to_object_id(id_str: str) -> ObjectId:
        return ObjectId(id_str)
