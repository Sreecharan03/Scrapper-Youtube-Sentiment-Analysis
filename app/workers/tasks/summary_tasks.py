"""
app/workers/tasks/summary_tasks.py
====================================
Celery task: generate LLM summary for a video transcript.

FLOW:
  1. Load transcript text from MongoDB (async)
  2. Mark summary as GENERATING (async)
  3. Call SummaryService.generate() — SYNCHRONOUS (Anthropic SDK is sync)
     This runs 2 Claude Haiku API calls with prompt caching.
  4. Store result in MongoDB (async)

PREREQUISITES:
  - Transcript must already be in status=completed (fetch_transcript ran first).
    If transcript is missing or incomplete, the task fails permanently.

COST:
  ~$0.012 per video (2 Haiku calls with transcript caching).
  Critique severity logged so you can monitor quality over time.
"""

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.scraper.pipeline import make_db_client
from app.services.llm_summary import SummaryService
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    bind                = True,
    name                = "generate_summary",
    queue               = "scraper",
    max_retries         = 2,
    default_retry_delay = 30,
    soft_time_limit     = 180,   # 3 min — 2 Haiku calls should finish in <60s
    time_limit          = 210,
    acks_late           = True,
    ignore_result       = True,
)
def generate_summary(self, *, video_id: str) -> dict:
    """
    Generate and store a structured LLM summary for the given video's transcript.

    Requires:
        - Transcript must be completed (fetch_transcript ran first)
        - ANTHROPIC_API_KEY must be set in .env
    """
    logger.info("summary_task_started", video_id=video_id, task_id=self.request.id)

    try:
        # ── Load transcript from MongoDB ──────────────────────────────────
        transcript_doc = asyncio.run(_get_transcript(video_id))

        if transcript_doc is None:
            raise ValueError(
                f"No transcript found for video {video_id!r}. "
                "Run fetch_transcript first."
            )
        if transcript_doc.get("status") != "completed":
            raise ValueError(
                f"Transcript for video {video_id!r} is not completed "
                f"(status={transcript_doc.get('status')!r}). "
                "Cannot generate summary from incomplete transcript."
            )

        # Build plain-text transcript from segments
        segments = transcript_doc.get("original_segments") or []
        if not segments:
            raise ValueError(f"Transcript for {video_id!r} has no segments.")

        transcript_text  = " ".join(s["text"] for s in segments)
        duration_secs    = transcript_doc.get("total_duration_secs", 0.0)

        # Pull video title from videos collection if available
        video_title = asyncio.run(_get_video_title(video_id))

        # ── Mark as GENERATING ────────────────────────────────────────────
        asyncio.run(_mark_generating(video_id))

        # ── Run LLM (synchronous — Anthropic SDK) ─────────────────────────
        settings = get_settings()
        service  = SummaryService(api_key=settings.anthropic_api_key)

        result = service.generate(
            transcript_text = transcript_text,
            duration_secs   = duration_secs,
            video_title     = video_title,
            segments        = segments,   # pass raw segments for accurate timestamps
        )

        # ── Store result ──────────────────────────────────────────────────
        asyncio.run(_store_completed(video_id, result))

        logger.info(
            "summary_task_completed",
            video_id           = video_id,
            critique_severity  = result.get("_meta", {}).get("critique_severity"),
            total_input_tokens = result.get("_meta", {}).get("total_input_tokens"),
        )
        return {"video_id": video_id, "status": "completed"}

    except ValueError as exc:
        # Permanent — bad input, no retry
        logger.error("summary_task_permanent_failure", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}

    except Exception as exc:
        if self.request.retries < self.max_retries:
            backoff = 30 * (2 ** self.request.retries)
            logger.warning(
                "summary_task_retrying",
                video_id = video_id,
                attempt  = self.request.retries + 1,
                backoff  = backoff,
                error    = str(exc),
            )
            raise self.retry(exc=exc, countdown=backoff)

        logger.error("summary_task_failed", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}


# ── Async DB helpers ──────────────────────────────────────────────────────

async def _get_transcript(video_id: str) -> dict | None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.transcript_repo import TranscriptRepository
        return await TranscriptRepository(db).get_transcript(video_id)
    finally:
        mongo_client.close()


async def _get_video_title(video_id: str) -> str | None:
    mongo_client, db = make_db_client()
    try:
        doc = await db["videos"].find_one({"video_id": video_id}, {"title": 1})
        return doc.get("title") if doc else None
    finally:
        mongo_client.close()


async def _mark_generating(video_id: str) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.summary_repo import SummaryRepository
        await SummaryRepository(db).mark_generating(video_id)
    finally:
        mongo_client.close()


async def _store_completed(video_id: str, data: dict) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.summary_repo import SummaryRepository
        await SummaryRepository(db).mark_completed(video_id, data)
    finally:
        mongo_client.close()


async def _mark_failed(video_id: str, error: str) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.summary_repo import SummaryRepository
        await SummaryRepository(db).mark_failed(video_id, error)
    finally:
        mongo_client.close()
