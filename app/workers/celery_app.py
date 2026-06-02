"""
app/workers/celery_app.py
==========================
Celery application instance and configuration.

WHY SEPARATE FROM main.py:
  - Celery runs as its OWN process: `celery -A app.workers.celery_app worker`
  - It imports this file, not main.py.
  - FastAPI also imports this to call `.delay()` on tasks.
  - Keeping it separate means both processes share the same Celery config
    without either importing the other's startup logic.

REDIS DB SEPARATION (from app/core/config.py):
  - Cache  (DB 0): Application cache — video metadata, rate limits, sessions
  - Broker (DB 1): Celery task messages
  - Result (DB 2): Celery task results
  URLs are built automatically from REDIS_HOST/PORT/USERNAME/PASSWORD in .env.
  Celery uses its own internal sync connection — separate from the async
  redis.asyncio client in app/core/redis_client.py.

STARTING THE WORKER (from project root):
  celery -A app.workers.celery_app worker --loglevel=info --concurrency=4
"""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "yt_scraper",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.tasks.scrape_tasks",      # Phase 1 stub (kept for compat)
        "app.workers.tasks.job_tasks",         # scrape_job_start, finalize_job
        "app.workers.tasks.tlc_tasks",         # scrape_tlc_batch (self-chaining)
        "app.workers.tasks.reply_tasks",       # scrape_reply_batch (reply pool)
        "app.workers.tasks.transcript_tasks",  # fetch_transcript (Phase 3A)
        "app.workers.tasks.summary_tasks",     # generate_summary (Phase 3A LLM)
    ],
)

# ------------------------------------------------------------------ #
# Celery configuration                                                 #
# ------------------------------------------------------------------ #
celery_app.conf.update(
    # Suppress Celery 6.0 deprecation warning for broker retry on startup
    broker_connection_retry_on_startup=True,

    # ── Redis connection pool limits ───────────────────────────────────────
    # Redis Cloud free tier hard limit: 30 connections (confirmed via INFO).
    # ALL tasks use ignore_result=True — no result backend writes at all.
    # This eliminates per-ForkPoolWorker result connections (the main culprit).
    #
    # Budget with single combined worker (concurrency=6) + FastAPI:
    #   worker base (broker + consumer + control + heartbeat): ~7
    #   per-task pipeline.make_redis_client (c=6 × 1):          6
    #   FastAPI app pool:                                        3
    #   Total:                                                  ~16  (14 under limit)
    broker_pool_limit=2,          # kombu Redis broker connection pool
    redis_max_connections=2,      # result backend pool (minimal — results ignored)

    # Serialization — JSON is human-readable and language-agnostic
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Result expiry — don't let Redis fill up with stale results
    result_expires=3600 * 24,  # 24 hours

    # Retry behaviour
    task_acks_late=True,        # Ack AFTER task completes (safer — no lost tasks on crash)
    task_reject_on_worker_lost=True,  # Re-queue if worker dies mid-task

    # Concurrency control per worker process
    worker_prefetch_multiplier=1,   # Don't pre-fetch — prevents one worker hoarding tasks
    task_max_retries=3,

    # ── Queue routing ──────────────────────────────────────────────────────
    # Three queues:
    #   scraper  — TLC batches + job lifecycle (long-running, few workers)
    #   replies  — reply pool workers (short tasks, high concurrency)
    #   default  — housekeeping / periodic tasks
    #
    # Launch TLC workers:   celery -A app.workers.celery_app worker -Q scraper --concurrency=2
    # Launch reply workers: celery -A app.workers.celery_app worker -Q replies --concurrency=8
    task_routes={
        "app.workers.tasks.scrape_tasks.*": {"queue": "scraper"},
        "scrape_job_start":                 {"queue": "scraper"},
        "scrape_tlc_batch":                 {"queue": "scraper"},
        "finalize_job":                     {"queue": "scraper"},
        "scrape_reply_batch":               {"queue": "replies"},
        "fetch_transcript":                 {"queue": "scraper"},
        "generate_summary":                 {"queue": "scraper"},
    },

    # Default queue for tasks without explicit routing
    task_default_queue="default",
)
