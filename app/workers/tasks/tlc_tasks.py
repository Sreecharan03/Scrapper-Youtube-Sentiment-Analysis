"""
app/workers/tasks/tlc_tasks.py
================================
Self-chaining TLC (top-level comment) batch worker.

HOW SELF-CHAINING WORKS:
  Each call to scrape_tlc_batch processes up to BATCH_SIZE (5 000) TLCs.
  On success, if there are more pages, it creates the NEXT batch document
  and calls scrape_tlc_batch.apply_async() for that next batch before
  returning.  This creates a chain:

    batch 1 → batch 2 → batch 3 → ... → batch N (exhausted)

  There is always exactly ONE active TLC task per job at a time.
  If any batch fails, the chain stops and the job is paused.

REPLY TOKEN DISPATCH:
  run_tlc_batch() pushes reply tokens to the Redis reply queue via
  cache.push_reply_tokens().  After the async run completes, the Celery task
  pops those tokens from the queue and dispatches individual
  scrape_reply_batch tasks — one per TLC that has replies.

FINALIZATION:
  When the last batch is exhausted (BatchResult.is_exhausted=True):
    1. Mark job.tlc_completed = True (MongoDB)
    2. Check reply_pending_count from Redis
    3. If == 0: all replies done too → fire finalize_job
    4. If > 0:  leave finalization to the last reply task

ERROR HANDLING:
  • ScraperVideoNotFoundError → permanent failure, job ends
  • ScraperRateLimitError     → pause job (operator must resume)
  • ScraperTimeoutError       → retry up to max_retries, then pause
  • DatabaseOperationError    → retry, then pause
  The batch document is updated on every state change so operators can
  see exactly which batch failed and why.
"""

import asyncio
import socket
from datetime import datetime, timezone

from app.core.logging import get_logger

from app.core.cache import CacheManager
from app.core.exceptions import (
    DatabaseOperationError,
    ScraperRateLimitError,
    ScraperTimeoutError,
    ScraperVideoNotFoundError,
)
from app.db.repositories.job_repo import JobRepository
from app.db.repositories.scrape_batch_repo import ScrapeBatchRepository
from app.models.job import JobStatus
from app.models.scrape_batch import ScrapeBatchDocument
from app.scraper.pipeline import make_db_client, make_redis_client, run_tlc_batch
from app.scraper.session import InnertubeContext
from app.workers.celery_app import celery_app
from app.workers.tasks.job_tasks import _context_from_dict, _context_to_dict, _fail_permanent

logger = get_logger(__name__)


# ── Task ──────────────────────────────────────────────────────────────────

@celery_app.task(
    bind             = True,
    name             = "scrape_tlc_batch",
    queue            = "scraper",
    max_retries      = 3,
    default_retry_delay = 60,
    soft_time_limit  = 1200,   # 20 min per batch (5000 TLCs × ~200 ms)
    time_limit       = 1260,
    acks_late        = True,
)
def scrape_tlc_batch(
    self,
    *,
    job_id:       str,
    video_id:     str,
    batch_id:     str,
    batch_number: int,
    start_token:  str,
    context:      dict,          # serialized InnertubeContext
) -> dict:
    """
    Scrape up to BATCH_SIZE top-level comments, then self-chain to next batch.
    """
    logger.info(
        "tlc_batch_started",
        job_id       = job_id,
        video_id     = video_id,
        batch_number = batch_number,
        task_id      = self.request.id,
    )

    try:
        result_dict = asyncio.run(
            _run_tlc_batch(
                job_id       = job_id,
                video_id     = video_id,
                batch_id     = batch_id,
                batch_number = batch_number,
                start_token  = start_token,
                context      = _context_from_dict(context),
                task_id      = self.request.id,
                worker       = socket.gethostname(),
            )
        )
        return result_dict

    except ScraperVideoNotFoundError as exc:
        _fail_permanent(job_id, video_id, str(exc))
        _pause_batch(job_id, batch_id, str(exc), permanent=True)
        return {"job_id": job_id, "status": "failed_permanent"}

    except ScraperRateLimitError as exc:
        _pause_batch(job_id, batch_id, str(exc))
        asyncio.run(_mark_job_paused_batch(job_id, batch_number, str(exc)))
        return {"job_id": job_id, "status": "paused_rate_limited"}

    except (ScraperTimeoutError, DatabaseOperationError) as exc:
        if self.request.retries < self.max_retries:
            backoff = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
            logger.warning(
                "tlc_batch_retrying",
                job_id=job_id, batch_number=batch_number,
                attempt=self.request.retries + 1, backoff=backoff,
            )
            raise self.retry(exc=exc, countdown=backoff)
        # All retries exhausted → pause
        _pause_batch(job_id, batch_id, str(exc))
        asyncio.run(_mark_job_paused_batch(job_id, batch_number, str(exc)))
        return {"job_id": job_id, "status": "paused_retries_exhausted"}

    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=120)
        _pause_batch(job_id, batch_id, f"Unexpected error: {exc}")
        asyncio.run(_mark_job_paused_batch(job_id, batch_number, str(exc)))
        return {"job_id": job_id, "status": "paused_unexpected_error"}


# ── Async core ─────────────────────────────────────────────────────────────

async def _run_tlc_batch(
    *,
    job_id:       str,
    video_id:     str,
    batch_id:     str,
    batch_number: int,
    start_token:  str,
    context:      InnertubeContext,
    task_id:      str,
    worker:       str,
) -> dict:
    """
    Full async execution for one TLC batch:
      1. Mark batch as RUNNING
      2. Run the batch scraper
      3. Pop reply tokens from Redis queue → dispatch reply tasks
      4. Update job progress counters
      5. Mark batch COMPLETED
      6. Self-chain or finalize
    """
    mongo_client, db = make_db_client()
    redis_client     = make_redis_client()
    try:
        cache      = CacheManager(redis_client=redis_client)
        job_repo   = JobRepository(db)
        batch_repo = ScrapeBatchRepository(db)

        # ── Guard: check job is still active ──────────────────────────────
        job = await job_repo.get_job(job_id)
        if job is None or job.get("status") not in JobStatus.ACTIVE_STATUSES:
            logger.warning(
                "tlc_batch_job_not_active",
                job_id=job_id, status=job.get("status") if job else "missing",
            )
            return {"job_id": job_id, "status": "skipped_inactive"}

        # ── Mark batch RUNNING ────────────────────────────────────────────
        await batch_repo.mark_running(batch_id, task_id, worker)

        # ── Run the batch ─────────────────────────────────────────────────
        result = await run_tlc_batch(
            job_id       = job_id,
            video_id     = video_id,
            batch_id     = batch_id,
            batch_number = batch_number,
            start_token  = start_token,
            context      = context,
            db           = db,
            cache        = cache,
        )

        # ── Dispatch reply tasks for tokens found in this batch ───────────
        reply_tasks_fired = 0
        if result.reply_tokens_found > 0:
            from app.workers.tasks.reply_tasks import scrape_reply_batch  # lazy import
            for _ in range(result.reply_tokens_found):
                token_data = await cache.pop_reply_token(job_id)
                if token_data is None:
                    break
                scrape_reply_batch.apply_async(
                    kwargs={
                        "job_id":      job_id,
                        "video_id":    video_id,
                        "comment_id":  token_data["comment_id"],
                        "reply_token": token_data["reply_token"],
                        "context":     _context_to_dict(context),
                    },
                    queue = "replies",
                )
                reply_tasks_fired += 1

        # ── Mark batch COMPLETED ──────────────────────────────────────────
        await batch_repo.mark_completed(
            batch_id,
            token_at_end = result.token_at_end or start_token,
            stats = {
                "comments_written":   result.comments_written,
                "duplicates_skipped": result.duplicates_skipped,
                "reply_tokens_found": result.reply_tokens_found,
                "sub_batches_done":   result.sub_batches_done,
            },
        )

        # ── Update job progress ───────────────────────────────────────────
        await job_repo.increment_comments_collected(job_id, result.comments_written)
        await job_repo.set_batch_progress(
            job_id,
            current_batch  = batch_number,
            total_completed = batch_number,   # sequential, so current == completed
        )
        if result.reply_tokens_found > 0:
            await job_repo.increment_reply_tokens_found(job_id, result.reply_tokens_found)

        logger.info(
            "tlc_batch_completed",
            job_id            = job_id,
            batch_number      = batch_number,
            comments_written  = result.comments_written,
            duplicates        = result.duplicates_skipped,
            reply_tasks_fired = reply_tasks_fired,
            is_exhausted      = result.is_exhausted,
        )

        # ── Self-chain or finalize ────────────────────────────────────────
        if not result.is_exhausted and result.next_token:
            # Create next batch document and fire next task
            next_batch_number = batch_number + 1
            next_batch_doc = ScrapeBatchDocument(
                job_id         = job_id,
                batch_number   = next_batch_number,
                token_at_start = result.next_token,
            )
            next_batch_id = await ScrapeBatchRepository(db).create_batch(next_batch_doc)

            from app.workers.tasks.tlc_tasks import scrape_tlc_batch
            scrape_tlc_batch.apply_async(
                kwargs={
                    "job_id":        job_id,
                    "video_id":      video_id,
                    "batch_id":      next_batch_id,
                    "batch_number":  next_batch_number,
                    "start_token":   result.next_token,
                    "context":       _context_to_dict(context),
                },
                queue = "scraper",
            )
        else:
            # TLC chain exhausted
            await job_repo.record_tlc_completed(job_id)
            pending = await cache.get_reply_pending_count(job_id)

            logger.info(
                "tlc_phase_exhausted",
                job_id        = job_id,
                batch_number  = batch_number,
                reply_pending = pending,
            )

            if pending == 0:
                # No replies either → finalize immediately
                from app.workers.tasks.job_tasks import finalize_job
                finalize_job.apply_async(
                    kwargs={"job_id": job_id, "video_id": video_id},
                    queue  = "scraper",
                    countdown = 2,   # small delay to let any in-flight reply tasks decrement
                )

        return {
            "job_id":           job_id,
            "batch_number":     batch_number,
            "comments_written": result.comments_written,
            "is_exhausted":     result.is_exhausted,
            "reply_tasks_fired":reply_tasks_fired,
        }

    finally:
        await redis_client.aclose()
        mongo_client.close()


# ── Error-path helpers ─────────────────────────────────────────────────────

def _pause_batch(batch_id: str, error: str, *, permanent: bool = False) -> None:
    """Synchronously mark a batch as failed/paused."""
    async def _run():
        mongo_client, db = make_db_client()
        try:
            repo = ScrapeBatchRepository(db)
            if permanent:
                await repo.mark_failed(batch_id, error)
            else:
                await repo.mark_paused(batch_id, error)
        finally:
            mongo_client.close()
    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.error("pause_batch_db_error", batch_id=batch_id, error=str(exc))


async def _mark_job_paused_batch(job_id: str, batch_number: int, error: str) -> None:
    """Async: transition job to PAUSED_BATCH_FAILED."""
    mongo_client, db = make_db_client()
    try:
        await JobRepository(db).mark_paused_batch_failed(job_id, batch_number, error)
    finally:
        mongo_client.close()
