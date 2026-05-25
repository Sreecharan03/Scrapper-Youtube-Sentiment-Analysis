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
        "app.workers.tasks.scrape_tasks",   # Phase 1 stub (kept for compat)
        "app.workers.tasks.job_tasks",      # scrape_job_start, finalize_job
        "app.workers.tasks.tlc_tasks",      # scrape_tlc_batch (self-chaining)
        "app.workers.tasks.reply_tasks",    # scrape_reply_batch (reply pool)
    ],
)

# ------------------------------------------------------------------ #
# Celery configuration                                                 #
# ------------------------------------------------------------------ #
celery_app.conf.update(
    # Suppress Celery 6.0 deprecation warning for broker retry on startup
    broker_connection_retry_on_startup=True,

    # ── Redis connection pool limits ───────────────────────────────────────
    # Managed Redis services have hard limits (e.g. 30-100 connections).
    # We cap every pool explicitly so the sum of all consumers stays safe:
    #   broker_pool_limit=3   → kombu broker transport pool
    #   redis_max_connections=5 → result backend pool
    #   FastAPI app pool      → max_connections=5 (redis_client.py)
    #   per-task client       → max_connections=2 (pipeline.make_redis_client)
    # Worst case with --concurrency=2:  3 + 5 + 5 + (2×2) = 17 connections
    broker_pool_limit=3,          # kombu Redis broker connection pool
    redis_max_connections=5,      # Celery result backend pool

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
    },

    # Default queue for tasks without explicit routing
    task_default_queue="default",
)
