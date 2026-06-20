"""
app/workers/tasks/clustering_tasks.py
=======================================
Celery task: Phase 3C BERTopic topic clustering.

FLOW:
  1. Guard: classification must be completed for this video
  2. Load all classified comments + video summary from MongoDB
  3. Check if current clustering is already fresh (< 10% new comments since last run)
  4. Mark cluster_info as processing
  5. Run ClusteringService.cluster() — full pipeline
  6. Write cluster docs to `clusters` collection (full replace)
  7. Write cluster_id field onto each comment doc in `comments` collection
  8. Mark cluster_info as completed with metadata

STALE DETECTION:
  If comment_count has grown by > 10% since last clustering, mark stale on GET.
  Re-clustering is triggered by POSTing again — not automatic (cluster IDs change).
"""

import asyncio
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories.cluster_repo import ClusterRepository, ClusterStatus
from app.db.repositories.comment_repo import CommentRepository
from app.scraper.pipeline import make_db_client
from app.services.clustering_service import ClusteringService
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    bind            = True,
    name            = "cluster_comments",
    queue           = "scraper",
    max_retries     = 1,
    soft_time_limit = 600,
    time_limit      = 660,
    acks_late       = True,
    ignore_result   = True,
)
def cluster_comments(self, *, video_id: str) -> dict:
    """
    Cluster all classified comments for a video into semantic topic groups.

    Prerequisite: POST /api/v1/analysis/{video_id}/classify must be completed first.
    """
    logger.info("clustering_task_started", video_id=video_id, task_id=self.request.id)

    try:
        result = asyncio.run(_run_clustering(video_id))
        logger.info("clustering_task_completed", video_id=video_id, **{
            k: v for k, v in result.items() if k != "video_id"
        })
        return result

    except ValueError as exc:
        logger.error("clustering_task_permanent_failure", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}

    except Exception as exc:
        if self.request.retries < self.max_retries:
            logger.warning(
                "clustering_task_retrying",
                video_id=video_id,
                attempt=self.request.retries + 1,
                error=str(exc),
            )
            raise self.retry(exc=exc, countdown=30, kwargs={"video_id": video_id})

        logger.error("clustering_task_failed", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}


# ── Core async logic ──────────────────────────────────────────────────────────

async def _run_clustering(video_id: str) -> dict:
    mongo_client, db = make_db_client()
    try:
        comment_repo = CommentRepository(db)
        cluster_repo = ClusterRepository(db)

        # ── Guard: classification must be completed ───────────────────────
        analysis = await db["comment_analysis"].find_one(
            {"video_id": video_id, "status": "completed"},
            {"_id": 0, "classified_count": 1},
        )
        if not analysis:
            raise ValueError(
                f"Classification not completed for {video_id!r}. "
                "Run POST /api/v1/analysis/{video_id}/classify first."
            )

        # ── Guard: don't re-cluster if already fresh ──────────────────────
        current_count = await comment_repo.count(
            {"video_id": video_id, "classification_status": "done"}
        )
        existing = await cluster_repo.get_info(video_id)
        if existing and existing.get("status") == ClusterStatus.COMPLETED:
            prev_count = existing.get("comment_count_at_cluster_time", 0)
            growth = (current_count - prev_count) / max(prev_count, 1)
            if growth <= 0.10:
                logger.info(
                    "clustering_already_fresh",
                    video_id=video_id,
                    prev_count=prev_count,
                    current_count=current_count,
                )
                return {
                    "video_id": video_id,
                    "status":   "already_fresh",
                    "message":  "Clusters are up to date. Growth < 10%.",
                }

        # ── Load summary ──────────────────────────────────────────────────
        summary = await db["summaries"].find_one(
            {"video_id": video_id, "status": "completed"},
            {"_id": 0},
        )
        if not summary:
            raise ValueError(
                f"No completed summary for {video_id!r}. "
                "Run POST /api/v1/summaries/{video_id} first."
            )

        # ── Load comments ─────────────────────────────────────────────────
        comments = await comment_repo.get_comments_for_clustering(video_id)
        if not comments:
            raise ValueError(f"No classified comments for {video_id!r}.")

        logger.info("clustering_comments_loaded", video_id=video_id, count=len(comments))

        # ── Mark processing ───────────────────────────────────────────────
        await cluster_repo.mark_processing(video_id, len(comments))

        # ── Run clustering pipeline ───────────────────────────────────────
        settings = get_settings()
        service  = ClusteringService(
            api_key = settings.groq_api_key,
            model   = settings.groq_model,
        )
        result = await service.cluster(comments, summary)

        if result.status == "skipped_insufficient_data":
            await cluster_repo._collection.update_one(
                {"video_id": video_id},
                {"$set": {
                    "status":       ClusterStatus.SKIPPED_INSUFFICIENT,
                    "completed_at": datetime.now(timezone.utc),
                }},
            )
            return {
                "video_id": video_id,
                "status":   "skipped_insufficient_data",
                "total_to_cluster": 0,
            }

        # ── Write cluster docs ────────────────────────────────────────────
        cluster_docs = [
            {
                "video_id":            video_id,
                "cluster_id":          cl.cluster_id,
                "label":               cl.label,
                "keywords":            cl.keywords,
                "label_confidence":    cl.label_confidence,
                "cluster_type":        cl.cluster_type,
                "comment_count":       cl.comment_count,
                "is_content_gap":      cl.is_content_gap,
                "gap_similarity_score": cl.gap_similarity_score,
                "intent_breakdown":    cl.intent_breakdown,
                "sentiment_breakdown": cl.sentiment_breakdown,
                "top_comments":        cl.top_comments,
                "clustering_version":  "v1",
                "created_at":          datetime.now(timezone.utc),
            }
            for cl in result.clusters
        ]
        await cluster_repo.replace_clusters(video_id, cluster_docs)

        # ── Write cluster_id onto comment docs ────────────────────────────
        await comment_repo.bulk_update_cluster_ids(video_id, result.comment_assignments)

        # ── Mark completed ────────────────────────────────────────────────
        content_gaps = [cl for cl in result.clusters if cl.is_content_gap]
        meta = {
            "video_id":                    video_id,
            "total_clusters":              len(result.clusters),
            "content_gap_count":           len(content_gaps),
            "comment_count_at_cluster_time": len(comments),
            "total_clustered":             result.total_clustered,
            "total_unclustered":           result.total_unclustered,
            "outlier_ratio_before":        result.outlier_ratio_before,
            "min_cluster_size_used":       result.min_cluster_size_used,
            "clustering_version":          "v1",
        }
        await cluster_repo.mark_completed(video_id, meta)

        return {
            "video_id":       video_id,
            "status":         "completed",
            "total_clusters": len(result.clusters),
            "content_gaps":   len(content_gaps),
            "clustered":      result.total_clustered,
            "unclustered":    result.total_unclustered,
        }

    finally:
        mongo_client.close()


async def _mark_failed(video_id: str, error: str) -> None:
    mongo_client, db = make_db_client()
    try:
        await ClusterRepository(db).mark_failed(video_id, error)
    finally:
        mongo_client.close()
