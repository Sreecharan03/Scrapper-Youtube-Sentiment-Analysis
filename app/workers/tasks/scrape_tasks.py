"""
app/workers/tasks/scrape_tasks.py
===================================
Celery task definitions for scraping operations.

PHASE 1 STATUS: Stubs only.
  The task signatures and retry/error-handling skeleton are defined here
  so that the FastAPI job endpoints can call `.delay()` immediately.
  The actual scraping logic (Phase 2) will be injected without changing
  these signatures.

HOW CELERY + ASYNC INTERACT:
  Celery workers run synchronously by default. Our scraper uses aiohttp
  (async). The pattern is:
    1. Celery task is sync (regular def).
    2. Inside the task, we create an asyncio event loop and run the async
       scraper in it: asyncio.run(async_scrape_function(...))
  This is the standard pattern — do NOT use celery-gevent or celery-eventlet
  unless you have a specific reason.
"""

import asyncio
from datetime import datetime, timezone

from celery import Task
from celery.utils.log import get_task_logger

from app.core.exceptions import RetryableError, ScraperError
from app.workers.celery_app import celery_app

logger = get_task_logger(__name__)


class ScrapeTaskBase(Task):
    """
    Base class for all scrape tasks.
    Provides shared on_failure / on_retry hooks so every task gets
    consistent error logging and job status updates without repeating code.
    """

    abstract = True  # Celery won't register this as a task itself

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when a task permanently fails (all retries exhausted)."""
        job_id = kwargs.get("job_id") or (args[0] if args else "unknown")
        logger.error(
            "scrape_task_failed_permanently",
            task_id=task_id,
            job_id=job_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        # Phase 2: call job_repo.mark_failed() here

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when a task is being retried."""
        job_id = kwargs.get("job_id") or (args[0] if args else "unknown")
        logger.warning(
            "scrape_task_retrying",
            task_id=task_id,
            job_id=job_id,
            error_type=type(exc).__name__,
            countdown=self.request.retries,
        )


# ------------------------------------------------------------------ #
# Task: scrape_video_comments                                          #
# ------------------------------------------------------------------ #

@celery_app.task(
    bind=True,                    # `self` = the Task instance (needed for retry)
    base=ScrapeTaskBase,
    name="scrape_video_comments", # Explicit name — avoids auto-name surprises
    queue="scraper",
    max_retries=3,
    default_retry_delay=60,       # Base retry delay in seconds (Celery applies backoff)
    soft_time_limit=600,          # Sends SIGTERM after 10 min — task can clean up
    time_limit=660,               # Sends SIGKILL after 11 min — hard stop
)
def scrape_video_comments(self, job_id: str, video_id: str, video_url: str) -> dict:
    """
    Main scrape task: collect all comments for a YouTube video.

    Args:
        job_id:    MongoDB _id of the Job document to track progress.
        video_id:  YouTube video ID (e.g. "dQw4w9WgXcQ").
        video_url: Full YouTube URL for the video.

    Returns:
        dict with keys: job_id, video_id, comments_collected, status

    Raises:
        RetryableError: Task will be retried with backoff.
        ScraperError:   Non-retryable failure, job marked as failed.
    """
    logger.info(
        "scrape_task_started",
        job_id=job_id,
        video_id=video_id,
        task_id=self.request.id,
    )

    try:
        # ---------------------------------------------------------- #
        # PHASE 2: Replace this stub with actual scraper call         #
        # result = asyncio.run(_run_scrape(job_id, video_id, video_url))
        # ---------------------------------------------------------- #
        result = _stub_scrape(job_id, video_id)
        logger.info("scrape_task_completed", **result)
        return result

    except RetryableError as exc:
        # Transient error — retry with exponential backoff
        backoff = 2 ** self.request.retries * 30   # 30s, 60s, 120s
        raise self.retry(exc=exc, countdown=backoff)

    except ScraperError as exc:
        # Non-retryable scraper error — fail permanently
        logger.error(
            "scrape_task_permanent_failure",
            job_id=job_id,
            error=str(exc),
        )
        raise   # Celery marks as FAILURE, triggers on_failure hook


def _stub_scrape(job_id: str, video_id: str) -> dict:
    """
    Placeholder implementation for Phase 1.
    Returns a fake success result so job flow can be tested end-to-end.
    Remove this when Phase 2 scraper is implemented.
    """
    logger.info("stub_scrape_running", job_id=job_id, video_id=video_id)
    return {
        "job_id": job_id,
        "video_id": video_id,
        "comments_collected": 0,
        "status": "completed",
        "note": "STUB — Phase 2 scraper not yet implemented",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ------------------------------------------------------------------ #
# Task: retry_failed_jobs                                              #
# ------------------------------------------------------------------ #

@celery_app.task(
    name="retry_failed_jobs",
    queue="default",
)
def retry_failed_jobs() -> dict:
    """
    Periodic task: find jobs stuck in FAILED with retries remaining
    and re-queue them.

    Run on a schedule (Phase 2: add to Celery beat schedule).
    Phase 1: stub that returns count of would-be-retried jobs.
    """
    logger.info("retry_failed_jobs_running")
    # Phase 2: query job_repo.get_retriable_jobs() and call scrape_video_comments.delay()
    return {"retried": 0, "note": "STUB — Phase 2 implementation pending"}
