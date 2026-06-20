"""
app/workers/tasks/recommendation_tasks.py
==========================================
Phase 3D Celery task: generate audience intelligence recommendations.

Dependencies:
  - Phase 3B (classification) must be completed
  - Phase 3C (clustering) must be completed
  - Phase 3A summary must exist

Flow:
  1. Guard checks — classify + cluster must be complete, summary must exist
  2. Load clusters, summary, misconception comments, unanswered questions
  3. RecommendationService.generate() → pure Python analysis + ONE Groq call
  4. Persist to recommendations collection
"""

import asyncio
from dataclasses import asdict

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories.cluster_repo import ClusterRepository, ClusterStatus
from app.db.repositories.comment_repo import CommentRepository
from app.db.repositories.recommendation_repo import RecommendationRepository
from app.db.repositories.summary_repo import SummaryRepository
from app.scraper.pipeline import make_db_client
from app.services.recommendation_service import RecommendationService
from app.workers.celery_app import celery_app

logger = get_logger(__name__)
settings = get_settings()


@celery_app.task(
    name="generate_recommendations",
    bind=True,
    soft_time_limit=300,
    ignore_result=True,
)
def generate_recommendations(self, video_id: str) -> None:
    asyncio.run(_generate_async(video_id))


async def _generate_async(video_id: str) -> None:
    mongo_client, db = make_db_client()
    try:
        cluster_repo = ClusterRepository(db)
        comment_repo = CommentRepository(db)
        summary_repo = SummaryRepository(db)
        rec_repo     = RecommendationRepository(db)

        # ── Guard: classification complete ────────────────────────────────
        cls_doc    = await db["comment_analysis"].find_one(
            {"video_id": video_id, "status": "completed"},
            {"_id": 0, "status": 1},
        )
        cls_status = cls_doc["status"] if cls_doc else None
        if cls_status != "completed":
            logger.warning(
                "recommendations_skipped_no_classification",
                video_id=video_id, cls_status=cls_status,
            )
            await rec_repo.mark_failed(
                video_id, f"Classification not completed (status={cls_status})"
            )
            return

        # ── Guard: clustering complete ────────────────────────────────────
        cluster_status = await cluster_repo.get_status(video_id)
        if cluster_status != ClusterStatus.COMPLETED:
            logger.warning(
                "recommendations_skipped_no_clusters",
                video_id=video_id, cluster_status=cluster_status,
            )
            await rec_repo.mark_failed(
                video_id, f"Clustering not completed (status={cluster_status})"
            )
            return

        # ── Guard: summary exists ─────────────────────────────────────────
        summary = await summary_repo.get_summary(video_id)
        if not summary:
            logger.warning("recommendations_skipped_no_summary", video_id=video_id)
            await rec_repo.mark_failed(video_id, "Summary not found")
            return

        await rec_repo.mark_processing(video_id)

        # ── Load data ─────────────────────────────────────────────────────
        clusters              = await cluster_repo.get_clusters(video_id)
        misconception_comments = await comment_repo.get_misconception_comments(video_id)
        unanswered_comments   = await comment_repo.get_unanswered_questions(video_id)

        logger.info(
            "recommendations_data_loaded",
            video_id=video_id,
            clusters=len(clusters),
            misconceptions=len(misconception_comments),
            unanswered=len(unanswered_comments),
        )

        # ── Generate ──────────────────────────────────────────────────────
        service = RecommendationService(
            api_key = settings.groq_api_key,
            model   = settings.groq_model,
        )
        result = await service.generate(
            clusters               = clusters,
            summary                = summary,
            misconception_comments = misconception_comments,
            unanswered_comments    = unanswered_comments,
        )

        # ── Persist ───────────────────────────────────────────────────────
        def _serialize_item(item) -> dict:
            d = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
            return d

        await rec_repo.mark_completed(video_id, {
            "content_gaps":         [_serialize_item(g) for g in result.content_gaps],
            "misconceptions":       [_serialize_item(m) for m in result.misconceptions],
            "controversy_hotspots": [_serialize_item(c) for c in result.controversy_hotspots],
            "unanswered_questions": [_serialize_item(q) for q in result.unanswered_questions],
        })

        logger.info(
            "recommendations_task_done",
            video_id=video_id,
            gaps=len(result.content_gaps),
            misconceptions=len(result.misconceptions),
            controversies=len(result.controversy_hotspots),
            unanswered=len(result.unanswered_questions),
        )

    except Exception as exc:
        logger.exception("recommendations_task_failed", video_id=video_id, error=str(exc))
        try:
            await rec_repo.mark_failed(video_id, str(exc))
        except Exception:
            pass
        raise
    finally:
        mongo_client.close()
