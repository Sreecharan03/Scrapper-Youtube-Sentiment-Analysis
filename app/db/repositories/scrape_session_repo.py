"""
app/db/repositories/scrape_session_repo.py
============================================
Repository for scrape_sessions — the durable continuation-token store.

This is the RECOVERY source.  If Redis is wiped, workers read from here
and rebuild Redis state before resuming scraping.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.repositories.base import BaseRepository
from app.models.scrape_session import ScrapeSessionDocument


class ScrapeSessionRepository(BaseRepository):
    collection_name = "scrape_sessions"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    async def create_session(self, session: ScrapeSessionDocument) -> str:
        return await self.insert_one(session.to_dict())

    async def get_session(self, job_id: str) -> Optional[dict]:
        return await self.find_one({"job_id": job_id})

    async def checkpoint(
        self,
        job_id: str,
        *,
        token: str,
        token_obtained_at: datetime,
        sub_batch_number: int,
        comments_written_total: int,
        current_batch_number: int,
    ) -> None:
        """
        Save the latest continuation token + progress counters.
        Called after every 100-comment sub-batch write succeeds.
        Uses upsert so it works whether or not the session already exists.
        """
        await self.update_one(
            {"job_id": job_id},
            {"$set": {
                "current_tlc_token":      token,
                "token_obtained_at":      token_obtained_at,
                "sub_batch_number":       sub_batch_number,
                "comments_written_total": comments_written_total,
                "current_batch_number":   current_batch_number,
                "last_checkpoint_at":     datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    async def delete_session(self, job_id: str) -> None:
        """Remove session on job completion — frees storage."""
        await self.delete_one({"job_id": job_id})
