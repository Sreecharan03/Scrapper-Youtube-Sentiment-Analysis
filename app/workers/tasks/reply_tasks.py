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
    soft_time_limit  = 600,   # 10 min — covers REPLY_TASK_BATCH_SIZE chains in parallel
    time_limit       = 660,
    acks_late        = True,
    ignore_result    = True,   # return value unused — finalization done via Redis counter
)
def scrape_reply_batch(
    self,
    *,
    job_id:  str,
    video_id: str,
    tokens:  list,   # list of {comment_id, reply_token} — REPLY_TASK_BATCH_SIZE per task
    context: dict,
) -> dict:
    """
    Fetch replies for a BATCH of TLC comments in parallel (asyncio.gather).

    Batching amortizes the expensive MongoDB Atlas connection setup across
    REPLY_TASK_BATCH_SIZE chains instead of paying it once per chain.
    asyncio.gather() runs all chains concurrently within a single event loop,
    so one Atlas connection serves all chains in the batch.
    """
    logger.info(
        "reply_batch_started",
        job_id     = job_id,
        batch_size = len(tokens),
        task_id    = self.request.id,
    )

    try:
        result = asyncio.run(
            _run_reply_batch_multi(
                job_id   = job_id,
                video_id = video_id,
                tokens   = tokens,
                context  = _context_from_dict(context),
                task_id  = self.request.id,
            )
        )
        return result

    except Exception as exc:
        if self.request.retries < self.max_retries:
            backoff = 30 * (2 ** self.request.retries)
            logger.warning(
                "reply_batch_retrying",
                job_id  = job_id,
                attempt = self.request.retries + 1,
                backoff = backoff,
                error   = str(exc),
            )
            raise self.retry(exc=exc, countdown=backoff)

        # All retries exhausted — mark all tokens as failed and release slots
        asyncio.run(_fail_all_tokens(job_id, video_id, tokens, error=str(exc)))
        return {"job_id": job_id, "status": "exhausted", "error": str(exc)}


# ── Async core ─────────────────────────────────────────────────────────────

async def _run_reply_batch_multi(
    *,
    job_id:   str,
    video_id: str,
    tokens:   list,   # [{comment_id, reply_token}, ...]
    context,
    task_id:  str,
) -> dict:
    """
    Process REPLY_TASK_BATCH_SIZE reply chains in parallel using asyncio.gather.

    One MongoDB + Redis connection pair is shared across all chains in the batch.
    This eliminates repeated Atlas topology-discovery overhead — the most
    expensive part of per-chain task execution (~5-10s per discovery).

    Each chain's reply_pending slot is decremented individually so the
    finalization counter stays accurate regardless of partial failures.
    """
    import random
    # Stagger Motor client creation across concurrent tasks.
    # Without this, all 6 workers call make_db_client() simultaneously →
    # 18 simultaneous Atlas replica-set handshakes → Atlas M0 rate-limits them
    # → all 6 get ReplicaSetNoPrimary after 30s.
    # A 0–3s jitter spreads the connections so Atlas handles them one at a time.
    await asyncio.sleep(random.uniform(0, 3))

    mongo_client, db = make_db_client()
    redis_client     = make_redis_client()
    try:
        cache    = CacheManager(redis_client=redis_client)
        job_repo = JobRepository(db)

        # ── Single guard check covers all tokens in this batch ────────────
        job = await job_repo.get_job(job_id)
        if job is None or job.get("status") not in (
            JobStatus.ACTIVE_STATUSES | {JobStatus.FINALIZING}
        ):
            # Job cancelled — release all pending slots atomically
            for _ in tokens:
                await cache.decrement_reply_pending(job_id)
            logger.info("reply_batch_skipped_inactive", job_id=job_id,
                        batch_size=len(tokens))
            return {"job_id": job_id, "status": "skipped_inactive",
                    "count": len(tokens)}

        # ── Process all chains concurrently ───────────────────────────────
        async def _process_one(token_data: dict) -> dict:
            comment_id  = token_data["comment_id"]
            reply_token = token_data["reply_token"]
            try:
                result = await run_reply_batch(
                    job_id      = job_id,
                    video_id    = video_id,
                    comment_id  = comment_id,
                    reply_token = reply_token,
                    context     = context,
                    db          = db,
                    cache       = cache,
                )
                if result.comments_written > 0:
                    await job_repo.increment_comments_collected(
                        job_id, result.comments_written
                    )
                await job_repo.increment_reply_tokens_completed(job_id)
                await _finalize_reply_slot(
                    job_id, video_id, failed=False, db=db, cache=cache
                )
                return {"comment_id": comment_id,
                        "written": result.comments_written, "status": "ok"}

            except Exception as exc:
                logger.warning("reply_chain_failed", job_id=job_id,
                               comment_id=comment_id, error=str(exc))
                await _record_reply_failure(
                    job_id, video_id, comment_id, reply_token,
                    error=str(exc), error_type=type(exc).__name__, exhausted=True,
                    db=db,   # reuse existing Motor client — avoids new Atlas topology discovery
                )
                await _finalize_reply_slot(
                    job_id, video_id, failed=True, db=db, cache=cache
                )
                return {"comment_id": comment_id, "status": "failed",
                        "error": str(exc)}

        results = await asyncio.gather(
            *[_process_one(t) for t in tokens],
            return_exceptions=False,
        )

        total_written = sum(r.get("written", 0) for r in results)
        failed_count  = sum(1 for r in results if r.get("status") == "failed")

        logger.info(
            "reply_batch_multi_completed",
            job_id        = job_id,
            batch_size    = len(tokens),
            total_written = total_written,
            failed        = failed_count,
        )

        return {
            "job_id":        job_id,
            "batch_size":    len(tokens),
            "total_written": total_written,
            "failed":        failed_count,
            "status":        "completed",
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


# ── Batch failure helper ───────────────────────────────────────────────────

async def _fail_all_tokens(
    job_id:   str,
    video_id: str,
    tokens:   list,
    *,
    error: str,
) -> None:
    """On task-level failure, mark all tokens as failed and release their slots."""
    mongo_client, db = make_db_client()
    redis_client     = make_redis_client()
    try:
        cache = CacheManager(redis_client=redis_client)
        for t in tokens:
            try:
                await _record_reply_failure(
                    job_id, video_id,
                    t["comment_id"], t["reply_token"],
                    error=error, error_type="TaskExhausted", exhausted=True,
                    db=db,
                )
            except Exception:
                pass
            await _finalize_reply_slot(
                job_id, video_id, failed=True, db=db, cache=cache
            )
    finally:
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
    db = None,   # pass existing db to avoid opening a new Motor client per failure
) -> None:
    """Persist a reply failure to MongoDB failed_replies collection."""
    own_client = db is None
    if own_client:
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
        if own_client:
            mongo_client.close()
