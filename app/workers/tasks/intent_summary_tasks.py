"""
app/workers/tasks/intent_summary_tasks.py
==========================================
Phase 3E-pre Celery task: generate per-intent audience summaries.

Guards:
  - Classification must be completed
  - Cache: if already completed AND comment count unchanged (< 10% growth), skip
"""

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories.cluster_repo import ClusterRepository
from app.db.repositories.comment_repo import CommentRepository
from app.db.repositories.intent_summary_repo import IntentSummaryRepository
from app.db.repositories.summary_repo import SummaryRepository
from app.scraper.pipeline import make_db_client
from app.services.intent_summary_service import IntentSummaryService
from app.workers.celery_app import celery_app

logger    = get_logger(__name__)
settings  = get_settings()


@celery_app.task(
    name="generate_intent_summaries",
    bind=True,
    soft_time_limit=180,
    ignore_result=True,
)
def generate_intent_summaries(self, video_id: str) -> None:
    asyncio.run(_run(video_id))


async def _run(video_id: str) -> None:
    mongo_client, db = make_db_client()
    try:
        comment_repo      = CommentRepository(db)
        cluster_repo      = ClusterRepository(db)
        summary_repo      = SummaryRepository(db)
        intent_summary_repo = IntentSummaryRepository(db)

        # ── Guard: classification complete ────────────────────────────────
        cls_doc = await db["comment_analysis"].find_one(
            {"video_id": video_id, "status": "completed"},
            {"_id": 0, "classified_count": 1},
        )
        if not cls_doc:
            logger.warning("intent_summary_skipped_no_classification", video_id=video_id)
            await intent_summary_repo.mark_failed(
                video_id, "Classification not completed"
            )
            return

        # ── Cache check: skip if fresh ────────────────────────────────────
        current_count = await comment_repo.count(
            {"video_id": video_id, "classification_status": "done"}
        )
        existing = await intent_summary_repo.get(video_id)
        if existing and existing.get("status") == "completed":
            prev_count = existing.get("comment_count_at_gen", 0)
            growth = (current_count - prev_count) / max(prev_count, 1)
            if growth <= 0.10:
                logger.info(
                    "intent_summary_cache_hit",
                    video_id=video_id,
                    prev=prev_count,
                    current=current_count,
                )
                return

        await intent_summary_repo.mark_processing(video_id)

        # ── Load data ─────────────────────────────────────────────────────
        intent_counts = await comment_repo.get_intent_counts(video_id)
        top_comments  = {}
        from app.services.intent_summary_service import LLM_INTENTS
        for intent in LLM_INTENTS:
            top_comments[intent] = await comment_repo.get_top_comments_by_intent(
                video_id, intent, limit=8
            )

        clusters      = await cluster_repo.get_clusters(video_id)
        video_summary = await summary_repo.get_summary(video_id) or {}

        logger.info(
            "intent_summary_data_loaded",
            video_id=video_id,
            intents=list(intent_counts.keys()),
            total=sum(intent_counts.values()),
        )

        # ── Generate ──────────────────────────────────────────────────────
        service = IntentSummaryService(api_key=settings.anthropic_api_key)
        result  = await service.generate(
            video_id      = video_id,
            intent_counts  = intent_counts,
            top_comments   = top_comments,
            clusters       = clusters,
            video_summary  = video_summary,
        )

        await intent_summary_repo.mark_completed(video_id, current_count, result)

        logger.info(
            "intent_summary_task_done",
            video_id=video_id,
            intents=list(result.get("intent_summaries", {}).keys()),
        )

    except Exception as exc:
        logger.exception("intent_summary_task_failed", video_id=video_id, error=str(exc))
        try:
            await intent_summary_repo.mark_failed(video_id, str(exc))
        except Exception:
            pass
        raise
    finally:
        mongo_client.close()
