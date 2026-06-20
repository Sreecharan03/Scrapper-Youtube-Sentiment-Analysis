"""
app/db/repositories/comment_repo.py
=====================================
Repository for the `comments` collection.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.core.logging import get_logger
from app.db.repositories.base import BaseRepository
from app.models.comment import CommentDocument

logger = get_logger(__name__)


class CommentRepository(BaseRepository):
    collection_name = "comments"

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        super().__init__(database)

    # ------------------------------------------------------------------ #
    # Domain-specific queries                                              #
    # ------------------------------------------------------------------ #

    async def bulk_insert_comments(
        self, comments: list[CommentDocument]
    ) -> tuple[int, int]:
        """
        Insert a batch of comments, ignoring duplicates (by comment_id + video_id).

        The `comments` index is a unique compound index on (video_id, comment_id),
        so re-inserting the same comment on a re-scrape is silently skipped
        via `ordered=False` in insert_many.

        Returns:
            (inserted_count, duplicate_count): counts for monitoring/logging.
        """
        if not comments:
            return 0, 0

        docs = [c.to_dict() for c in comments]
        try:
            result = await self._collection.insert_many(docs, ordered=False)
            inserted = len(result.inserted_ids)
            duplicates = len(docs) - inserted
            logger.info(
                "comments_batch_inserted",
                inserted=inserted,
                duplicates=duplicates,
            )
            return inserted, duplicates
        except Exception as exc:
            # BulkWriteError contains partial results — extract what succeeded
            if hasattr(exc, "details"):
                inserted = exc.details.get("nInserted", 0)
                duplicates = len(docs) - inserted
                logger.warning(
                    "comments_batch_partial",
                    inserted=inserted,
                    duplicates=duplicates,
                    error=str(exc),
                )
                return inserted, duplicates
            raise

    async def get_comments_for_video(
        self,
        video_id: str,
        *,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "published_at",
        descending: bool = True,
    ) -> list[dict]:
        """
        Paginated retrieval of comments for a video.

        Args:
            sort_by: Field to sort by ("published_at", "like_count", "scraped_at")
            descending: True = newest/most-liked first
        """
        direction = DESCENDING if descending else ASCENDING
        return await self.find_many(
            {"video_id": video_id},
            skip=skip,
            limit=limit,
            sort=[(sort_by, direction)],
        )

    async def count_comments_for_video(self, video_id: str) -> int:
        """Count how many comments we have stored for a video."""
        return await self.count({"video_id": video_id})

    async def get_top_comments(
        self, video_id: str, *, limit: int = 10
    ) -> list[dict]:
        """Fetch the most-liked top-level comments for a video."""
        return await self.find_many(
            {"video_id": video_id, "is_reply": False},
            limit=limit,
            sort=[("like_count", DESCENDING)],
        )

    async def comment_exists(self, comment_id: str, video_id: str) -> bool:
        """Check for a specific comment — used to avoid duplicate scraping."""
        return await self.exists({"comment_id": comment_id, "video_id": video_id})

    async def get_all_for_classification(self, video_id: str) -> list[dict]:
        """
        Load all comments for a video for classification.
        Minimal projection — only fields needed by the classifier.
        Returns unbounded list (no pagination) — classification always needs all comments.
        """
        cursor = self._collection.find(
            {"video_id": video_id},
            {
                "comment_id":        1,
                "text":              1,
                "is_reply":          1,
                "parent_comment_id": 1,
                "_id":               0,
            },
        )
        return await cursor.to_list(None)

    async def get_comments_for_clustering(self, video_id: str) -> list[dict]:
        """
        Load all classified comments for Phase 3C clustering.
        Includes text_hash (dedup), like_count (reply filter), author_channel_id (bot detection),
        intent_labels + sentiment (breakdown computation), and parent_comment_id (assignment fallback).
        """
        cursor = self._collection.find(
            {"video_id": video_id, "classification_status": "done"},
            {
                "comment_id":        1,
                "text":              1,
                "text_hash":         1,
                "is_reply":          1,
                "parent_comment_id": 1,
                "like_count":        1,
                "author_name":       1,
                "author_channel_id": 1,
                "intent_labels":     1,
                "sentiment":         1,
                "answered_by_video": 1,
                "classification_status": 1,
                "_id":               0,
            },
        )
        return await cursor.to_list(None)

    async def bulk_update_cluster_ids(
        self,
        video_id:    str,
        assignments: dict,  # {comment_id: cluster_id}
    ) -> int:
        """Bulk-write cluster_id field onto each comment doc."""
        from pymongo import UpdateOne

        if not assignments:
            return 0

        ops = [
            UpdateOne(
                {"comment_id": cid, "video_id": video_id},
                {"$set": {"cluster_id": cluster_id}},
            )
            for cid, cluster_id in assignments.items()
        ]
        result = await self._collection.bulk_write(ops, ordered=False)
        logger.info(
            "cluster_ids_written",
            video_id=video_id,
            modified=result.modified_count,
        )
        return result.modified_count

    async def get_failed_for_classification(self, video_id: str) -> list[dict]:
        """Load only comments that failed classification — for retry runs."""
        cursor = self._collection.find(
            {"video_id": video_id, "classification_status": "failed"},
            {
                "comment_id":        1,
                "text":              1,
                "is_reply":          1,
                "parent_comment_id": 1,
                "_id":               0,
            },
        )
        return await cursor.to_list(None)

    async def get_classification_counts(self, video_id: str) -> dict:
        """Count comments by classification_status for aggregate recomputation."""
        pipeline = [
            {"$match": {"video_id": video_id, "classification_status": "done"}},
            {"$group": {
                "_id": None,
                "sentiments":     {"$push": "$sentiment"},
                "intent_labels":  {"$push": "$intent_labels"},
                "total_done":     {"$sum": 1},
            }},
        ]
        results = await self._collection.aggregate(pipeline).to_list(1)
        return results[0] if results else {}

    async def bulk_update_classifications(
        self,
        video_id: str,
        results: list[dict],
    ) -> int:
        """
        Bulk update comment docs with classification results using unordered bulk_write.
        Unordered = maximum throughput; individual failures don't stop the batch.

        Returns:
            Number of documents modified.
        """
        from pymongo import UpdateOne

        if not results:
            return 0

        ops = []
        for r in results:
            cid    = r["comment_id"]
            status = r.get("classification_status", "done")

            if status == "done":
                set_fields: dict = {
                    "intent_labels":             r.get("intent_labels", []),
                    "sentiment":                 r.get("sentiment", "neutral"),
                    "classification_confidence": r.get("classification_confidence", 0.0),
                    "classification_status":     "done",
                    "classified_at":             r.get("classified_at"),
                    "classification_version":    r.get("classification_version", "v1"),
                }
                if "answered_by_video" in r:
                    set_fields["answered_by_video"] = r["answered_by_video"]
            else:
                set_fields = {
                    "classification_status":  status,
                    "classified_at":          r.get("classified_at"),
                    "classification_version": r.get("classification_version", "v1"),
                }

            ops.append(UpdateOne(
                {"comment_id": cid, "video_id": video_id},
                {"$set": set_fields},
            ))

        result = await self._collection.bulk_write(ops, ordered=False)
        logger.info(
            "comments_classifications_bulk_written",
            video_id=video_id,
            total_ops=len(ops),
            modified=result.modified_count,
        )
        return result.modified_count

    async def get_misconception_comments(self, video_id: str) -> list[dict]:
        """Load all classified misconception-labeled comments with cluster assignment."""
        cursor = self._collection.find(
            {
                "video_id":              video_id,
                "classification_status": "done",
                "intent_labels":         "misconception",
            },
            {
                "comment_id": 1,
                "text":       1,
                "like_count": 1,
                "cluster_id": 1,
                "_id":        0,
            },
        )
        return await cursor.to_list(None)

    async def get_unanswered_questions(self, video_id: str) -> list[dict]:
        """Load unanswered question comments sorted by like_count descending."""
        cursor = self._collection.find(
            {
                "video_id":              video_id,
                "classification_status": "done",
                "intent_labels":         "question",
                "answered_by_video":     False,
            },
            {
                "comment_id": 1,
                "text":       1,
                "like_count": 1,
                "cluster_id": 1,
                "_id":        0,
            },
        ).sort("like_count", -1)
        return await cursor.to_list(None)

    async def get_top_comments_by_intent(
        self,
        video_id: str,
        intent:   str,
        limit:    int = 8,
    ) -> list[dict]:
        """Top comments for a given intent label, sorted by like_count."""
        cursor = self._collection.find(
            {
                "video_id":              video_id,
                "classification_status": "done",
                "intent_labels":         intent,
            },
            {"comment_id": 1, "text": 1, "like_count": 1, "cluster_id": 1, "_id": 0},
        ).sort("like_count", -1).limit(limit)
        return await cursor.to_list(None)

    async def get_intent_counts(self, video_id: str) -> dict:
        """Count of comments per intent label for a video."""
        pipeline = [
            {"$match": {"video_id": video_id, "classification_status": "done"}},
            {"$unwind": "$intent_labels"},
            {"$group": {"_id": "$intent_labels", "count": {"$sum": 1}}},
        ]
        rows = await self._collection.aggregate(pipeline).to_list(None)
        return {row["_id"]: row["count"] for row in rows}
