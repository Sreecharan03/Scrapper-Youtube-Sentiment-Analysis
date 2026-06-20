"""
app/db/repositories/cluster_repo.py
=====================================
Repository for the `clusters` and `cluster_info` collections.

clusters     — one doc per (video_id, cluster_id). Replaces all on each run.
cluster_info — one doc per video_id. Tracks clustering job status + metadata.
"""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository

logger = get_logger(__name__)


class ClusterStatus:
    PROCESSING              = "processing"
    COMPLETED               = "completed"
    FAILED                  = "failed"
    SKIPPED_INSUFFICIENT    = "skipped_insufficient_data"


class ClusterRepository(BaseRepository):
    """Manages cluster_info (job state) for the clustering pipeline."""

    collection_name = "cluster_info"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)
        self._clusters_col = database["clusters"]

    # ── Job state ────────────────────────────────────────────────────────────

    async def get_info(self, video_id: str) -> Optional[dict]:
        return await self.find_one({"video_id": video_id})

    async def get_status(self, video_id: str) -> Optional[str]:
        doc = await self._collection.find_one(
            {"video_id": video_id}, {"status": 1}
        )
        return doc.get("status") if doc else None

    async def mark_processing(self, video_id: str, comment_count: int) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                "video_id":    video_id,
                "status":      ClusterStatus.PROCESSING,
                "started_at":  datetime.now(timezone.utc),
                "comment_count_at_cluster_time": comment_count,
                "error":       None,
            }},
            upsert=True,
        )
        logger.info("clustering_marked_processing", video_id=video_id)

    async def mark_completed(self, video_id: str, meta: dict) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                **meta,
                "status":       ClusterStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "stale":        False,
                "error":        None,
            }},
            upsert=True,
        )
        logger.info(
            "clustering_marked_completed",
            video_id=video_id,
            total_clusters=meta.get("total_clusters"),
            content_gaps=meta.get("content_gap_count"),
        )

    async def mark_failed(self, video_id: str, error: str) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {
                "status": ClusterStatus.FAILED,
                "error":  error,
            }},
            upsert=True,
        )
        logger.error("clustering_marked_failed", video_id=video_id, error=error)

    async def mark_stale(self, video_id: str) -> None:
        await self._collection.update_one(
            {"video_id": video_id},
            {"$set": {"stale": True}},
        )

    # ── Cluster docs ─────────────────────────────────────────────────────────

    async def replace_clusters(self, video_id: str, cluster_docs: list[dict]) -> None:
        """
        Full replace — delete all existing clusters for this video, then insert fresh.
        Called at the end of each successful clustering run.
        """
        await self._clusters_col.delete_many({"video_id": video_id})
        if cluster_docs:
            await self._clusters_col.insert_many(cluster_docs, ordered=False)
        logger.info(
            "clusters_replaced",
            video_id=video_id, count=len(cluster_docs),
        )

    async def get_clusters(self, video_id: str) -> list[dict]:
        """Return all clusters for a video, sorted by comment_count DESC."""
        cursor = self._clusters_col.find(
            {"video_id": video_id},
            {"_id": 0},
        ).sort([("comment_count", -1)])
        return await cursor.to_list(None)

    async def get_cluster(self, video_id: str, cluster_id: int) -> Optional[dict]:
        return await self._clusters_col.find_one(
            {"video_id": video_id, "cluster_id": cluster_id},
            {"_id": 0},
        )
