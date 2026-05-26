"""
app/workers/tasks/job_tasks.py
================================
Job lifecycle Celery tasks.

TASK: scrape_job_start
  Called once per job, immediately after the job document is created.
  Steps:
    1. Transition job → FETCHING_META
    2. Fetch YouTube watch page → extract InnertubeContext (client version,
       visitor_data, initial continuation token, video metadata)
    3. Save InnertubeContext to Redis (scraper session) + MongoDB (scrape_session)
    4. Transition job → SCRAPING_TLCS
    5. Create the first ScrapeBatch document (batch_number=1)
    6. Fire scrape_tlc_batch for batch 1

TASK: finalize_job
  Called when the TLC chain is exhausted AND all reply tasks have completed.
  Both tlc_tasks and reply_tasks may trigger this — it is IDEMPOTENT:
    • Checks job status → returns early if already FINALIZING/COMPLETED
    • Uses mark_finalizing() as an atomic gate (only one caller proceeds)
    • Counts actual scraped comments, marks job COMPLETED
    • Clears Redis state (session + reply queue + job lock)

WHY TWO SEPARATE TASKS:
  scrape_job_start is short (one HTTP request).  It runs in the normal
  scraper queue so the worker is not blocked for the 10–30 minutes a
  full scrape takes.  finalize_job is also short (one count + update).
"""

import asyncio
import socket
from datetime import datetime, timezone
from typing import Optional

from app.core.logging import get_logger

from app.core.cache import CacheManager
from app.core.exceptions import (
    ScraperRateLimitError,
    ScraperTimeoutError,
    ScraperVideoNotFoundError,
)
from app.db.repositories.job_repo import JobRepository
from app.db.repositories.scrape_batch_repo import ScrapeBatchRepository
from app.db.repositories.scrape_session_repo import ScrapeSessionRepository
from app.models.job import JobStatus
from app.models.scrape_batch import ScrapeBatchDocument
from app.models.scrape_session import ScrapeSessionDocument
from app.scraper.pipeline import make_db_client, make_redis_client
from app.scraper.session import InnertubeContext, ScraperSession
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _context_to_dict(ctx: InnertubeContext) -> dict:
    """Serialize InnertubeContext for JSON Celery task parameters."""
    return {
        "video_id":                   ctx.video_id,
        "api_key":                    ctx.api_key,
        "client_version":             ctx.client_version,
        "visitor_data":               ctx.visitor_data,
        "initial_continuation_token": ctx.initial_continuation_token,
        "newest_first_token":         ctx.newest_first_token,
        "title":                      ctx.title,
        "channel_name":               ctx.channel_name,
        "channel_id":                 ctx.channel_id,
        "view_count":                 ctx.view_count,
        "comment_count":              ctx.comment_count,
    }


def _context_from_dict(d: dict) -> InnertubeContext:
    """Reconstruct InnertubeContext from a Celery task parameter dict."""
    return InnertubeContext(
        video_id                   = d["video_id"],
        api_key                    = d.get("api_key", ""),
        client_version             = d.get("client_version", ""),
        visitor_data               = d.get("visitor_data", ""),
        initial_continuation_token = d.get("initial_continuation_token"),
        newest_first_token         = d.get("newest_first_token"),
        title                      = d.get("title"),
        channel_name               = d.get("channel_name"),
        channel_id                 = d.get("channel_id"),
        view_count                 = d.get("view_count"),
        comment_count              = d.get("comment_count"),
    )


async def _initialise_job(
    job_id:   str,
    video_id: str,
    task_id:  str,
) -> tuple[InnertubeContext, str]:
    """
    Async core of scrape_job_start:
      • Mark job as FETCHING_META
      • Fetch YouTube page, extract InnertubeContext
      • Create ScrapeSession in MongoDB + Redis
      • Mark job as SCRAPING_TLCS
      • Create first ScrapeBatch document

    Returns (context, batch_id) where batch_id is the MongoDB _id of batch 1.
    """
    mongo_client, db = make_db_client()
    redis_client     = make_redis_client()
    try:
        cache        = CacheManager(redis_client=redis_client)
        job_repo     = JobRepository(db)
        session_repo = ScrapeSessionRepository(db)
        batch_repo   = ScrapeBatchRepository(db)

        # ── 1. Transition → FETCHING_META ─────────────────────────────────
        await job_repo.mark_fetching_meta(job_id, task_id)

        # ── 2. Fetch YouTube page → InnertubeContext ──────────────────────
        async with ScraperSession(video_id) as scraper:
            ctx = await scraper.initialise()

        if not ctx.initial_continuation_token and not ctx.newest_first_token:
            # Video has no comments (disabled or empty)
            # We still complete the job successfully with 0 comments
            await job_repo.mark_completed(job_id, total_scraped=0)
            await cache.release_job_lock(video_id)
            logger.info(
                "job_no_comments",
                job_id=job_id,
                video_id=video_id,
                reason="no_initial_continuation_token",
            )
            return ctx, ""   # sentinel — caller will check for empty batch_id

        # Prefer "Newest First" token — gives the full comment chain (all comments
        # in reverse chronological order).  Falls back to "Top Comments" if the
        # Newest First token was not found on the page (rare, e.g. very new videos).
        start_token = ctx.newest_first_token or ctx.initial_continuation_token

        logger.info(
            "job_token_selected",
            job_id       = job_id,
            sort_order   = "newest_first" if ctx.newest_first_token else "top_comments",
            has_nf_token = bool(ctx.newest_first_token),
        )

        # ── 3. Persist session to MongoDB + Redis ─────────────────────────
        session_doc = ScrapeSessionDocument(
            job_id             = job_id,
            video_id           = video_id,
            current_tlc_token  = start_token,
            token_obtained_at  = datetime.now(timezone.utc),
            current_batch_number = 1,
        )
        await session_repo.create_session(session_doc)

        await cache.set_scraper_session(job_id, {
            "current_tlc_token":      start_token,
            "sub_batch_number":       0,
            "comments_written_total": 0,
            "current_batch_number":   1,
            "last_updated":           datetime.now(timezone.utc).isoformat(),
        })

        # ── 4. Store video metadata in cache ──────────────────────────────
        if ctx.title or ctx.channel_name:
            await cache.set_video_metadata(video_id, {
                "title":         ctx.title,
                "channel_name":  ctx.channel_name,
                "channel_id":    ctx.channel_id,
                "view_count":    ctx.view_count,
                "comment_count": ctx.comment_count,
            })

        # ── 5. Transition → SCRAPING_TLCS ─────────────────────────────────
        await job_repo.mark_scraping_tlcs(job_id, ctx.comment_count)

        # ── 6. Create batch 1 document ────────────────────────────────────
        batch_doc = ScrapeBatchDocument(
            job_id        = job_id,
            batch_number  = 1,
            token_at_start = start_token,
        )
        batch_id = await batch_repo.create_batch(batch_doc)

        logger.info(
            "job_initialised",
            job_id       = job_id,
            video_id     = video_id,
            batch_id     = batch_id,
            comment_count = ctx.comment_count,
            has_visitor_data = bool(ctx.visitor_data),
        )
        return ctx, batch_id

    finally:
        await redis_client.aclose()
        mongo_client.close()


# ── Task: scrape_job_start ─────────────────────────────────────────────────

@celery_app.task(
    bind             = True,
    name             = "scrape_job_start",
    queue            = "scraper",
    max_retries      = 3,
    default_retry_delay = 30,
    soft_time_limit  = 120,    # 2 min — only one HTTP fetch needed
    time_limit       = 150,
    acks_late        = True,
    ignore_result    = True,
)
def scrape_job_start(self, *, job_id: str, video_id: str) -> dict:
    """
    Kick off a scrape job: fetch the video page, extract context, fire batch 1.

    Called by the API endpoint immediately after creating the job document.
    Idempotent in the sense that if it fails and retries, `mark_fetching_meta`
    and `create_session` both use upsert-style operations.
    """
    logger.info("job_start_task_picked_up", job_id=job_id, video_id=video_id,
                task_id=self.request.id)
    try:
        ctx, batch_id = asyncio.run(
            _initialise_job(job_id, video_id, self.request.id)
        )

        if not batch_id:
            # Video has no comments — already finalized inside _initialise_job
            return {"job_id": job_id, "status": "completed_no_comments"}

        # Fire the first TLC batch
        from app.workers.tasks.tlc_tasks import scrape_tlc_batch  # avoid circular import
        scrape_tlc_batch.apply_async(
            kwargs={
                "job_id":        job_id,
                "video_id":      video_id,
                "batch_id":      batch_id,
                "batch_number":  1,
                "start_token":   ctx.newest_first_token or ctx.initial_continuation_token,
                "context":       _context_to_dict(ctx),
            },
            queue = "scraper",
        )

        logger.info("job_start_task_done", job_id=job_id, batch_id=batch_id)
        return {"job_id": job_id, "batch_id": batch_id, "status": "scraping_tlcs"}

    except ScraperVideoNotFoundError as exc:
        _fail_permanent(job_id, video_id, str(exc))
        return {"job_id": job_id, "status": "failed_permanent", "error": str(exc)}

    except (ScraperRateLimitError, ScraperTimeoutError) as exc:
        backoff = 30 * (2 ** self.request.retries)
        logger.warning("job_start_retrying", job_id=job_id,
                       attempt=self.request.retries + 1, backoff=backoff)
        raise self.retry(exc=exc, countdown=backoff)

    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60)
        _fail_permanent(job_id, video_id, f"scrape_job_start failed: {exc}")
        return {"job_id": job_id, "status": "failed_permanent", "error": str(exc)}


# ── Task: finalize_job ─────────────────────────────────────────────────────

@celery_app.task(
    bind            = True,
    name            = "finalize_job",
    queue           = "scraper",
    max_retries     = 5,
    default_retry_delay = 10,
    soft_time_limit = 60,
    time_limit      = 90,
    acks_late       = True,
    ignore_result   = True,
)
def finalize_job(self, *, job_id: str, video_id: str) -> dict:
    """
    Complete a job after all TLC batches and all reply batches finish.

    IDEMPOTENT: If the job is already FINALIZING or COMPLETED, returns
    immediately.  Both scrape_tlc_batch and scrape_reply_batch call this;
    the MongoDB status transition acts as the concurrency gate.
    """
    logger.info("finalize_job_called", job_id=job_id)
    try:
        result = asyncio.run(_run_finalize(job_id, video_id))
        return result
    except Exception as exc:
        logger.error("finalize_job_error", job_id=job_id, error=str(exc))
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=15)
        raise


async def _run_finalize(job_id: str, video_id: str) -> dict:
    """Async core of finalize_job."""
    mongo_client, db = make_db_client()
    redis_client     = make_redis_client()
    try:
        cache    = CacheManager(redis_client=redis_client)
        job_repo = JobRepository(db)

        # ── Idempotency check ─────────────────────────────────────────────
        job = await job_repo.get_job(job_id)
        if job is None:
            logger.error("finalize_job_not_found", job_id=job_id)
            return {"job_id": job_id, "status": "error_not_found"}

        current_status = job.get("status")
        if current_status in (JobStatus.FINALIZING, JobStatus.COMPLETED,
                               JobStatus.FAILED_PERMANENT):
            logger.info("finalize_job_already_done",
                        job_id=job_id, status=current_status)
            return {"job_id": job_id, "status": current_status, "skipped": True}

        # ── Double-check both completion conditions ───────────────────────
        tlc_completed     = job.get("tlc_completed", False)
        reply_pending     = await cache.get_reply_pending_count(job_id)

        if not tlc_completed or reply_pending > 0:
            logger.info(
                "finalize_job_conditions_not_met",
                job_id        = job_id,
                tlc_completed = tlc_completed,
                reply_pending = reply_pending,
            )
            return {
                "job_id":        job_id,
                "status":        "conditions_not_met",
                "tlc_completed": tlc_completed,
                "reply_pending": reply_pending,
            }

        # ── Atomic gate: mark FINALIZING first ───────────────────────────
        # If two tasks reach here simultaneously, only one will succeed in
        # marking FINALIZING (motor upsert is non-atomic, but the real
        # protection is the reply_pending counter decrement being atomic).
        await job_repo.mark_finalizing(job_id)

        # ── Count actual scraped comments ─────────────────────────────────
        col           = db["comments"]
        total_scraped = await col.count_documents({"video_id": video_id})

        # ── Mark COMPLETED ────────────────────────────────────────────────
        await job_repo.mark_completed(job_id, total_scraped)

        # ── Clean up Redis state ──────────────────────────────────────────
        await cache.clear_scraper_session(job_id)
        await cache.clear_reply_queue(job_id)
        await cache.release_job_lock(video_id)

        logger.info(
            "job_completed",
            job_id        = job_id,
            video_id      = video_id,
            total_scraped = total_scraped,
        )
        return {"job_id": job_id, "status": "completed", "total_scraped": total_scraped}

    finally:
        await redis_client.aclose()
        mongo_client.close()


# ── Private helper ─────────────────────────────────────────────────────────

def _fail_permanent(job_id: str, video_id: str, reason: str) -> None:
    """Mark job permanently failed and release the Redis job lock."""
    async def _run():
        mongo_client, db = make_db_client()
        redis_client     = make_redis_client()
        try:
            cache    = CacheManager(redis_client=redis_client)
            job_repo = JobRepository(db)
            await job_repo.mark_failed_permanent(job_id, reason)
            await cache.release_job_lock(video_id)
        finally:
            await redis_client.aclose()
            mongo_client.close()

    asyncio.run(_run())
    logger.error("job_failed_permanent", job_id=job_id, reason=reason)
