"""
app/db/init_db.py
==================
Database initialisation — creates all collections and indexes on startup.
Idempotent: safe to call on every startup, existing indexes are no-ops.

INDEX STRATEGY SUMMARY:
  videos          → video_id (unique)
  comments        → (video_id, comment_id) unique | video_id | like sort | time sort
  jobs            → (video_id, status) | status+created | created_at
  scrape_batches  → (job_id, batch_number) unique | job_id+status
  scrape_sessions → job_id (unique)
  comment_history → (comment_id, version) unique | video_id+detected_at
  failed_replies  → (job_id, comment_id) unique | status+created
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Collection name constants ──────────────────────────────────────────────
VIDEOS_COLLECTION          = "videos"
COMMENTS_COLLECTION        = "comments"
JOBS_COLLECTION            = "jobs"
SCRAPE_BATCHES_COLLECTION  = "scrape_batches"
SCRAPE_SESSIONS_COLLECTION = "scrape_sessions"
COMMENT_HISTORY_COLLECTION = "comment_history"
FAILED_REPLIES_COLLECTION  = "failed_replies"
TRANSCRIPTS_COLLECTION       = "transcripts"
SUMMARIES_COLLECTION         = "summaries"
COMMENT_ANALYSIS_COLLECTION  = "comment_analysis"
CLUSTERS_COLLECTION          = "clusters"
CLUSTER_INFO_COLLECTION      = "cluster_info"
RECOMMENDATIONS_COLLECTION   = "recommendations"
INTENT_SUMMARIES_COLLECTION  = "intent_summaries"

ALL_COLLECTIONS = [
    VIDEOS_COLLECTION,
    COMMENTS_COLLECTION,
    JOBS_COLLECTION,
    SCRAPE_BATCHES_COLLECTION,
    SCRAPE_SESSIONS_COLLECTION,
    COMMENT_HISTORY_COLLECTION,
    FAILED_REPLIES_COLLECTION,
    TRANSCRIPTS_COLLECTION,
    SUMMARIES_COLLECTION,
    COMMENT_ANALYSIS_COLLECTION,
    CLUSTERS_COLLECTION,
    CLUSTER_INFO_COLLECTION,
    RECOMMENDATIONS_COLLECTION,
    INTENT_SUMMARIES_COLLECTION,
]


async def init_db(database: AsyncIOMotorDatabase) -> None:
    """
    Create all collections and indexes.
    Call once at application startup and once in Celery worker startup.
    """
    logger.info("db_init_started", database=database.name)

    await _init_videos(database)
    await _init_comments(database)
    await _init_jobs(database)
    await _init_scrape_batches(database)
    await _init_scrape_sessions(database)
    await _init_comment_history(database)
    await _init_failed_replies(database)
    await _init_transcripts(database)
    await _init_summaries(database)
    await _init_comment_analysis(database)
    await _init_clusters(database)
    await _init_cluster_info(database)
    await _init_recommendations(database)
    await _init_intent_summaries(database)

    logger.info("db_init_completed", database=database.name)


# ── Per-collection initializers ────────────────────────────────────────────

async def _init_videos(db: AsyncIOMotorDatabase) -> None:
    col = db[VIDEOS_COLLECTION]
    await col.create_index(
        [("video_id", ASCENDING)],
        unique=True, name="idx_video_id_unique", background=True,
    )
    await col.create_index(
        [("scrape_completed", ASCENDING), ("created_at", DESCENDING)],
        name="idx_incomplete_scrapes", background=True,
    )
    logger.debug("indexes_ready", collection=VIDEOS_COLLECTION)


async def _safe_create_index(col, keys, *, name: str, **kwargs) -> None:
    """
    Create an index, automatically handling the case where a stale index
    exists with the same name but different keys (OperationFailure code 86).
    In that case, drop the old index and recreate it.
    """
    try:
        await col.create_index(keys, name=name, **kwargs)
    except OperationFailure as exc:
        if exc.code == 86:   # IndexKeySpecsConflict
            logger.warning(
                "index_key_conflict_dropping_old",
                collection=col.name, index=name, error=str(exc),
            )
            await col.drop_index(name)
            await col.create_index(keys, name=name, **kwargs)
        else:
            raise


async def _init_comments(db: AsyncIOMotorDatabase) -> None:
    col = db[COMMENTS_COLLECTION]

    # Primary deduplication key — prevents duplicate comments on re-scrape.
    # insert_many with ordered=False silently skips violations.
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("comment_id", ASCENDING)],
        unique=True, name="idx_video_comment_unique", background=True,
    )
    # Most common query: "get all comments for this video"
    await _safe_create_index(
        col,
        [("video_id", ASCENDING)],
        name="idx_video_id", background=True,
    )
    # "Top comments" sort
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("like_count", DESCENDING)],
        name="idx_video_likes", background=True,
    )
    # Chronological feed (uses published_at_approx from Phase 2 schema)
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("published_at_approx", DESCENDING)],
        name="idx_video_published", background=True,
    )
    # TLC-only queries (exclude replies)
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("is_reply", ASCENDING)],
        name="idx_video_is_reply", background=True,
    )
    # Re-scrape edit detection: find comments by hash change
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("text_hash", ASCENDING)],
        name="idx_video_text_hash", background=True,
    )
    # Status filter for "not_visible" recovery scans
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("status", ASCENDING)],
        name="idx_video_status", background=True,
    )
    # Phase 3B: filter by intent label — "get all questions for a video"
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("intent_labels", ASCENDING)],
        name="idx_video_intent_labels", background=True,
    )
    # Phase 3B: filter by classification status — resume incomplete runs
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("classification_status", ASCENDING)],
        name="idx_video_classification_status", background=True,
    )
    logger.debug("indexes_ready", collection=COMMENTS_COLLECTION)


async def _init_jobs(db: AsyncIOMotorDatabase) -> None:
    col = db[JOBS_COLLECTION]
    await col.create_index(
        [("video_id", ASCENDING), ("status", ASCENDING)],
        name="idx_video_status", background=True,
    )
    await col.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)],
        name="idx_status_created", background=True,
    )
    await col.create_index(
        [("created_at", DESCENDING)],
        name="idx_created_at", background=True,
    )
    logger.debug("indexes_ready", collection=JOBS_COLLECTION)


async def _init_scrape_batches(db: AsyncIOMotorDatabase) -> None:
    col = db[SCRAPE_BATCHES_COLLECTION]
    # Unique per job+batch_number — prevents accidental duplicate batch docs
    await col.create_index(
        [("job_id", ASCENDING), ("batch_number", ASCENDING)],
        unique=True, name="idx_job_batch_unique", background=True,
    )
    # "Find all running batches for a job" — used by health-check task
    await col.create_index(
        [("job_id", ASCENDING), ("status", ASCENDING)],
        name="idx_job_status", background=True,
    )
    logger.debug("indexes_ready", collection=SCRAPE_BATCHES_COLLECTION)


async def _init_scrape_sessions(db: AsyncIOMotorDatabase) -> None:
    col = db[SCRAPE_SESSIONS_COLLECTION]
    # One session per job — used by recovery logic
    await col.create_index(
        [("job_id", ASCENDING)],
        unique=True, name="idx_job_id_unique", background=True,
    )
    logger.debug("indexes_ready", collection=SCRAPE_SESSIONS_COLLECTION)


async def _init_comment_history(db: AsyncIOMotorDatabase) -> None:
    col = db[COMMENT_HISTORY_COLLECTION]
    # Each (comment_id, version) pair is unique
    await col.create_index(
        [("comment_id", ASCENDING), ("version", ASCENDING)],
        unique=True, name="idx_comment_version_unique", background=True,
    )
    # "Show all edits for a video sorted by detection time"
    await col.create_index(
        [("video_id", ASCENDING), ("detected_at", DESCENDING)],
        name="idx_video_detected_at", background=True,
    )
    logger.debug("indexes_ready", collection=COMMENT_HISTORY_COLLECTION)


async def _init_transcripts(db: AsyncIOMotorDatabase) -> None:
    col = db[TRANSCRIPTS_COLLECTION]
    # One transcript per video — video_id is the primary lookup key
    await col.create_index(
        [("video_id", ASCENDING)],
        unique=True, name="idx_transcript_video_unique", background=True,
    )
    # Filter by status (e.g. "find all failed fetches for retry")
    await col.create_index(
        [("status", ASCENDING)],
        name="idx_transcript_status", background=True,
    )
    logger.debug("indexes_ready", collection=TRANSCRIPTS_COLLECTION)


async def _init_summaries(db: AsyncIOMotorDatabase) -> None:
    col = db[SUMMARIES_COLLECTION]
    await col.create_index(
        [("video_id", ASCENDING)],
        unique=True, name="idx_summary_video_unique", background=True,
    )
    await col.create_index(
        [("status", ASCENDING)],
        name="idx_summary_status", background=True,
    )
    logger.debug("indexes_ready", collection=SUMMARIES_COLLECTION)


async def _init_comment_analysis(db: AsyncIOMotorDatabase) -> None:
    col = db[COMMENT_ANALYSIS_COLLECTION]
    # One document per video — video_id is the primary lookup key
    await _safe_create_index(
        col,
        [("video_id", ASCENDING)],
        unique=True, name="idx_comment_analysis_video_unique", background=True,
    )
    # Filter by status — "find all processing jobs for monitoring"
    await _safe_create_index(
        col,
        [("status", ASCENDING)],
        name="idx_comment_analysis_status", background=True,
    )
    logger.debug("indexes_ready", collection=COMMENT_ANALYSIS_COLLECTION)


async def _init_clusters(db: AsyncIOMotorDatabase) -> None:
    col = db[CLUSTERS_COLLECTION]
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("cluster_id", ASCENDING)],
        unique=True, name="idx_clusters_video_cluster_unique", background=True,
    )
    await _safe_create_index(
        col,
        [("video_id", ASCENDING), ("is_content_gap", ASCENDING)],
        name="idx_clusters_gap_filter", background=True,
    )
    logger.debug("indexes_ready", collection=CLUSTERS_COLLECTION)


async def _init_cluster_info(db: AsyncIOMotorDatabase) -> None:
    col = db[CLUSTER_INFO_COLLECTION]
    await _safe_create_index(
        col,
        [("video_id", ASCENDING)],
        unique=True, name="idx_cluster_info_video_unique", background=True,
    )
    logger.debug("indexes_ready", collection=CLUSTER_INFO_COLLECTION)


async def _init_intent_summaries(db: AsyncIOMotorDatabase) -> None:
    col = db[INTENT_SUMMARIES_COLLECTION]
    await _safe_create_index(
        col,
        [("video_id", ASCENDING)],
        unique=True, name="idx_intent_summaries_video_unique", background=True,
    )
    await _safe_create_index(
        col,
        [("status", ASCENDING)],
        name="idx_intent_summaries_status", background=True,
    )
    logger.debug("indexes_ready", collection=INTENT_SUMMARIES_COLLECTION)


async def _init_recommendations(db: AsyncIOMotorDatabase) -> None:
    col = db[RECOMMENDATIONS_COLLECTION]
    await _safe_create_index(
        col,
        [("video_id", ASCENDING)],
        unique=True, name="idx_recommendations_video_unique", background=True,
    )
    await _safe_create_index(
        col,
        [("status", ASCENDING)],
        name="idx_recommendations_status", background=True,
    )
    logger.debug("indexes_ready", collection=RECOMMENDATIONS_COLLECTION)


async def _init_failed_replies(db: AsyncIOMotorDatabase) -> None:
    col = db[FAILED_REPLIES_COLLECTION]
    # One failed-reply record per (job, comment) pair
    await col.create_index(
        [("job_id", ASCENDING), ("comment_id", ASCENDING)],
        unique=True, name="idx_job_comment_unique", background=True,
    )
    # Recovery query: find all pending_retry records across jobs
    await col.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)],
        name="idx_status_created", background=True,
    )
    logger.debug("indexes_ready", collection=FAILED_REPLIES_COLLECTION)
