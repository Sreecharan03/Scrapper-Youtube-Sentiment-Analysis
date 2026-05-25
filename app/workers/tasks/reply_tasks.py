"""
app/workers/tasks/reply_tasks.py
==================================
Reply batch worker — processes replies for a single top-level comment (TLC).

HOW IT FITS IN THE PIPELINE:
  1. TLC batch worker (tlc_tasks.py) dispatches one scrape_reply_batch task
     per TLC that has replies.
  2. Each task calls run_reply_batch(), which fetches ALL reply pages for
     one TLC until exhausted (there is no sub-batching for replies —
     they're fetched in a single chain of continuation pages).
  3. After processing, the task decrements the reply_pending counter in Redis.
  4. When the counter reaches 0 AND job.tlc_completed is True, the LAST
     reply task to finish triggers finalize_job.

PARALLELISM:
  Multiple reply tasks run concurrently on the "replies" Celery queue.
  Workers processing replies should be separate from the main scraper
  queue to avoid reply tasks starving TLC batches.
  Recommended worker launch:
      celery -A app.workers.celery_app worker -Q replies --concurrency=8

ERROR HANDLING:
  Reply failures are non-fatal to the job.  If a reply batch fails after
  all retries:
    • The failure is recorded in the failed_replies collection
    • The reply_pending counter is still decremented (the slot is consumed)
    • job.reply_tokens_failed is incremented
  This ensures the job can always reach COMPLETED even if some replies
  are unrecoverable (e.g., reply continuation tokens that expired).

FINALIZATION RACE SAFETY:
  Both TLC completion and the last reply task may simultaneously think
  they should trigger finalize_job.  This is safe because finalize_job
  itself checks both conditions atomically before proceeding, and uses
  mark_finalizing() as a soft gate.
"""

import asyncio
import socket
from datetime import datetime, timezone

from app.core.logging import get_logger

from app.core.cache import CacheManager
from app.core.exceptions import ScraperVideoNotFoundError
from app.db.repositories.failed_reply_repo import FailedReplyRepository
from app.db.repositories.job_repo import JobRepository
from app.models.failed_reply import FailedReplyDocument
from app.models.job import JobStatus
from app.scraper.pipeline import make_db_client, make_redis_client, run_reply_batch
from app.workers.celery_app import celery_app
from app.workers.tasks.job_tasks import _context_from_dict

logger = get_logger(__name__)


# ── Task ──────────────────────────────────────────────────────────────────

@celery_app.task(
    bind             = True,
    name             = "scrape_reply_batch",
    queue            = "replies",
    max_retries      = 3,
    default_retry_delay = 30,
    soft_time_limit  = 300,   # 5 min per TLC reply chain (generous)
    time_limit       = 360,
    acks_late        = True,
)
def scrape_reply_batch(
    self,
    *,
    job_id:      str,
    video_id:    str,
    comment_id:  str,    # TLC whose replies we're fetching
    reply_token: str,
    context:     dict,   # serialized InnertubeContext
) -> dict:
    """
    Fetch ALL replies for one TLC comment and write them to MongoDB.
    """
    logger.info(
        "reply_batch_started",
        job_id     = job_id,
        comment_id = comment_id,
        task_id    = self.request.id,
    )

    try:
        result = asyncio.run(
            _run_reply_batch(
                job_id      = job_id,
                video_id    = video_id,
                comment_id  = comment_id,
                reply_token = reply_token,
                context     = _context_from_dict(context),
                task_id     = self.request.id,
            )
        )
        return result

    except ScraperVideoNotFoundError as exc:
        # Replies are gone (video deleted mid-scrape) — record and move on
        asyncio.run(_record_reply_failure(
            job_id, video_id, comment_id, reply_token,
            error=str(exc), error_type="VideoNotFound",
        ))
        asyncio.run(_finalize_reply_slot(job_id, video_id, failed=True))
        return {"job_id": job_id, "comment_id": comment_id, "status": "failed_permanent"}

    except Exception as exc:
        if self.request.retries < self.max_retries:
            backoff = 30 * (2 ** self.request.retries)
            logger.warning(
                "reply_batch_retrying",
                job_id     = job_id,
                comment_id = comment_id,
                attempt    = self.request.retries + 1,
                backoff    = backoff,
            )
            raise self.retry(exc=exc, countdown=backoff)

        # All retries exhausted — record failure, release the pending slot
        asyncio.run(_record_reply_failure(
            job_id, video_id, comment_id, reply_token,
            error=str(exc), error_type=type(exc).__name__,
            exhausted=True,
        ))
        asyncio.run(_finalize_reply_slot(job_id, video_id, failed=True))
        return {
            "job_id":     job_id,
            "comment_id": comment_id,
            "status":     "exhausted",
            "error":      str(exc),
        }


# ── Async core ─────────────────────────────────────────────────────────────

async def _run_reply_batch(
    *,
    job_id:      str,
    video_id:    str,
    comment_id:  str,
    reply_token: str,
    context,
    task_id:     str,
) -> dict:
    """Async execution: fetch all replies, decrement counter, maybe finalize."""
    mongo_client, db = make_db_client()
    redis_client     = make_redis_client()
    try:
        cache    = CacheManager(redis_client=redis_client)
        job_repo = JobRepository(db)

        # ── Guard: check job still active ────────────────────────────────
        job = await job_repo.get_job(job_id)
        if job is None or job.get("status") not in (
            JobStatus.ACTIVE_STATUSES | {JobStatus.FINALIZING}
        ):
            # Job was cancelled/failed while this reply task was queued — skip
            await cache.decrement_reply_pending(job_id)
            return {"job_id": job_id, "comment_id": comment_id, "status": "skipped_inactive"}

        # ── Fetch all replies for this TLC ────────────────────────────────
        result = await run_reply_batch(
            job_id      = job_id,
            video_id    = video_id,
            comment_id  = comment_id,
            reply_token = reply_token,
            context     = context,
            db          = db,
            cache       = cache,
        )

        # ── Update job counters ───────────────────────────────────────────
        if result.comments_written > 0:
            await job_repo.increment_comments_collected(job_id, result.comments_written)
        await job_repo.increment_reply_tokens_completed(job_id)

        # ── Decrement reply pending + check finalization ──────────────────
        await _finalize_reply_slot(job_id, video_id, failed=False, db=db, cache=cache)

        logger.info(
            "reply_batch_completed",
            job_id           = job_id,
            comment_id       = comment_id,
            comments_written = result.comments_written,
            sub_batches_done = result.sub_batches_done,
        )

        return {
            "job_id":           job_id,
            "comment_id":       comment_id,
            "comments_written": result.comments_written,
            "status":           "completed",
        }

    finally:
        await redis_client.aclose()
        mongo_client.close()


# ── Finalization check ─────────────────────────────────────────────────────

async def _finalize_reply_slot(
    job_id:   str,
    video_id: str,
    *,
    failed:   bool,
    db=None,
    cache:    CacheManager = None,
) -> None:
    """
    Decrement the reply_pending counter.
    If it reaches 0 AND tlc_completed=True → trigger finalize_job.

    Can be called with existing db/cache (success path) or without
    (error path — creates its own connections).
    """
    own_connections = db is None
    if own_connections:
        mongo_client, db = make_db_client()
        redis_client     = make_redis_client()
        cache            = CacheManager(redis_client=redis_client)

    try:
        if failed:
            job_repo = JobRepository(db)
            await job_repo.increment_reply_tokens_failed(job_id)

        pending = await cache.decrement_reply_pending(job_id)

        if pending == 0:
            # Check if TLC phase is also done
            job_repo = JobRepository(db)
            job      = await job_repo.get_job(job_id)
            if job and job.get("tlc_completed"):
                from app.workers.tasks.job_tasks import finalize_job
                finalize_job.apply_async(
                    kwargs={"job_id": job_id, "video_id": video_id},
                    queue  = "scraper",
                    countdown = 1,
                )
                logger.info(
                    "finalize_triggered_by_reply",
                    job_id  = job_id,
                    pending = pending,
                )

    finally:
        if own_connections:
            await redis_client.aclose()
            mongo_client.close()


# ── Failure recording ──────────────────────────────────────────────────────

async def _record_reply_failure(
    job_id:     str,
    video_id:   str,
    comment_id: str,
    reply_token:str,
    *,
    error:      str,
    error_type: str,
    exhausted:  bool = False,
) -> None:
    """Persist a reply failure to MongoDB failed_replies collection."""
    mongo_client, db = make_db_client()
    try:
        repo = FailedReplyRepository(db)
        doc  = FailedReplyDocument(
            job_id      = job_id,
            video_id    = video_id,
            comment_id  = comment_id,
            reply_token = reply_token,
            last_error  = error,
            last_error_type = error_type,
        )
        await repo.record_failure(doc)
        if exhausted:
            await repo.mark_exhausted(job_id, comment_id)
    except Exception as exc:
        logger.error("record_reply_failure_db_error",
                     job_id=job_id, comment_id=comment_id, error=str(exc))
    finally:
        mongo_client.close()
