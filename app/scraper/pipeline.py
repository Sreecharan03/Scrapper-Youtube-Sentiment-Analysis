"""
app/scraper/pipeline.py
========================
The scraping pipeline — orchestrates fetcher, parser, and DB writes.

This module contains the core async functions called by Celery tasks.
Everything here is pure async — Celery tasks wrap these in asyncio.run().

KEY FUNCTION: run_tlc_batch()
  Fetches up to BATCH_SIZE top-level comments in SUB_BATCH_SIZE chunks.
  After each chunk:
    1. Writes to MongoDB (insert_many, ordered=False — silently skips dupes)
    2. Saves checkpoint to Redis + MongoDB
    3. Pushes reply tokens to Redis reply queue
  Returns a BatchResult with the next continuation token (or None if done).

KEY FUNCTION: run_reply_batch()
  Fetches ALL replies for one TLC until the reply continuation is exhausted.
  Same sub-batch write + checkpoint pattern.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.cache import CacheManager
from app.core.config import get_settings
from app.core.exceptions import (
    DatabaseOperationError, ScraperRateLimitError,
    ScraperTimeoutError, ScraperVideoNotFoundError,
)
from app.core.logging import get_logger
from app.db.repositories.comment_history_repo import CommentHistoryRepository
from app.db.repositories.scrape_batch_repo import ScrapeBatchRepository
from app.db.repositories.scrape_session_repo import ScrapeSessionRepository
from app.models.comment import CommentDocument, CommentStatus
from app.models.comment_history import CommentHistoryDocument
from app.scraper.constants import (
    BATCH_SIZE, PERMANENT_ERROR_CODES,
    RATE_LIMIT_BACKOFF, RATE_LIMIT_STATUS_CODES,
    SUB_BATCH_SIZE, YT_PAGE_SIZE,
)
from app.scraper.parser import ParsedPage, parse_continuation_response
from app.scraper.session import InnertubeContext, ScraperSession

logger = get_logger(__name__)


@dataclass
class BatchResult:
    """Returned by run_tlc_batch() and run_reply_batch()."""
    comments_written:    int  = 0
    duplicates_skipped:  int  = 0
    reply_tokens_found:  int  = 0
    sub_batches_done:    int  = 0
    next_token:          Optional[str] = None   # None = exhausted
    token_at_end:        Optional[str] = None
    is_exhausted:        bool = False


# ── Connection factory (used inside Celery tasks) ─────────────────────────

def make_db_client() -> tuple[AsyncIOMotorClient, AsyncIOMotorDatabase]:
    """
    Create a fresh Motor client for use inside a Celery task's asyncio.run().
    Each Celery task creates its own client and closes it on completion.
    """
    settings = get_settings()
    # TLS only for Atlas (mongodb+srv://) — local MongoDB doesn't use TLS
    use_tls = settings.mongodb_uri.startswith("mongodb+srv://")
    client_kwargs = dict(
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=20_000,
        socketTimeoutMS=60_000,
    )
    if use_tls:
        client_kwargs["tls"] = True
        client_kwargs["tlsAllowInvalidCertificates"] = settings.mongodb_tls_allow_invalid_certs

    client   = AsyncIOMotorClient(settings.mongodb_uri, **client_kwargs)
    return client, client[settings.mongodb_db_name]


def make_redis_client():
    """
    Create a fresh async Redis client for use inside asyncio.run().

    Uses BlockingConnectionPool(max_connections=1) so that concurrent coroutines
    within the same asyncio.gather() queue up on the single connection instead of
    raising "Too many connections".  Without blocking=True, the second coroutine
    to request the connection while the first holds it gets an immediate error.

    max_connections=1 keeps per-task Redis usage to exactly 1 connection regardless
    of how many chains are running in parallel (asyncio is single-threaded — only
    one coroutine actually executes at a time, so serialised access is correct).
    """
    import redis.asyncio as aioredis
    from redis.asyncio import BlockingConnectionPool
    s = get_settings()
    # Build kwargs — ssl handled separately because BlockingConnectionPool
    # does not accept ssl= directly; SSLConnection class would be needed for ssl=True.
    # Our Redis instance uses ssl=False so we simply omit it (plain TCP is default).
    pool_kwargs = dict(
        host                   = s.redis_host,
        port                   = s.redis_port,
        db                     = s.redis_cache_db,
        username               = s.redis_username,
        password               = s.redis_password or None,
        decode_responses       = True,
        socket_connect_timeout = 10,
        socket_timeout         = 10,
        max_connections        = 1,   # one real connection per task
        timeout                = 30,  # wait up to 30s for the connection to free up
    )
    if s.redis_ssl:
        from redis.asyncio.connection import SSLConnection
        pool_kwargs["connection_class"] = SSLConnection
    pool = BlockingConnectionPool(**pool_kwargs)
    return aioredis.Redis(connection_pool=pool)


# ── TLC Batch Runner ───────────────────────────────────────────────────────

async def run_tlc_batch(
    *,
    job_id:        str,
    video_id:      str,
    batch_id:      str,
    batch_number:  int,
    start_token:   str,
    context:       InnertubeContext,
    db:            AsyncIOMotorDatabase,
    cache:         CacheManager,
) -> BatchResult:
    """
    Scrape up to BATCH_SIZE (5 000) top-level comments.
    Writes in SUB_BATCH_SIZE (100) chunks with checkpoints.

    Returns BatchResult.  Caller (Celery task) decides what to do next.
    """
    result      = BatchResult()
    token       = start_token
    accumulator: list[CommentDocument] = []
    page_number = 0

    batch_repo   = ScrapeBatchRepository(db)
    session_repo = ScrapeSessionRepository(db)
    history_repo = CommentHistoryRepository(db)

    async with ScraperSession(video_id) as scraper:
        scraper.context = context

        while result.comments_written + len(accumulator) < BATCH_SIZE:
            # ── API call ────────────────────────────────────────────────
            try:
                raw = await _fetch_with_retry(scraper, token)
            except ScraperVideoNotFoundError:
                raise
            except ScraperRateLimitError:
                raise
            except Exception as exc:
                logger.error("tlc_api_call_failed", job_id=job_id, error=str(exc))
                raise ScraperTimeoutError(f"API call failed: {exc}") from exc

            # ── Parse ────────────────────────────────────────────────────
            page = parse_continuation_response(raw, video_id, is_reply=False)
            page_number += 1

            if page.comments:
                for c in page.comments:
                    c.scrape_job_id      = job_id
                    c.scrape_batch_number = batch_number
                    c.scrape_page_number  = page_number
                accumulator.extend(page.comments)

            # Collect reply tokens
            result.reply_tokens_found += len(page.reply_tokens)
            if page.reply_tokens:
                await cache.push_reply_tokens(job_id, page.reply_tokens)

            # Advance token
            if page.next_token:
                token = page.next_token

            result.token_at_end = token

            # ── Write sub-batch when full ────────────────────────────────
            if len(accumulator) >= SUB_BATCH_SIZE or page.is_last_page:
                if accumulator:
                    written, dupes = await _write_sub_batch(
                        accumulator, db, history_repo, job_id
                    )
                    result.comments_written  += written
                    result.duplicates_skipped += dupes
                    result.sub_batches_done   += 1
                    accumulator.clear()

                    # ── Checkpoint after every successful write ──────────
                    # write_session_checkpoint only on the first sub-batch
                    # of each batch — the session doc is for resume only and
                    # doesn't need to be updated on every 300-comment chunk.
                    await _checkpoint(
                        job_id                   = job_id,
                        batch_id                 = batch_id,
                        batch_number             = batch_number,
                        token                    = token,
                        result                   = result,
                        cache                    = cache,
                        session_repo             = session_repo,
                        batch_repo               = batch_repo,
                        context                  = context,
                        write_session_checkpoint = (result.sub_batches_done == 1),
                    )

            # ── Page exhausted check ─────────────────────────────────────
            if page.is_last_page:
                result.is_exhausted = True
                result.next_token   = None
                break

            result.next_token = token

            # Delay between TLC API calls (see constants.REQUEST_DELAY_MIN/MAX_MS)
            await scraper.random_delay()

    return result


# ── Reply Batch Runner ─────────────────────────────────────────────────────

async def run_reply_batch(
    *,
    job_id:      str,
    video_id:    str,
    comment_id:  str,   # TLC whose replies we're fetching
    reply_token: str,
    context:     InnertubeContext,
    db:          AsyncIOMotorDatabase,
    cache:       CacheManager,
) -> BatchResult:
    """
    Fetch ALL replies for one TLC (follows continuation until exhausted).
    """
    result      = BatchResult()
    token       = reply_token
    accumulator: list[CommentDocument] = []
    history_repo = CommentHistoryRepository(db)

    async with ScraperSession(video_id) as scraper:
        scraper.context = context

        while True:
            try:
                raw = await _fetch_with_retry(scraper, token)
            except Exception as exc:
                raise ScraperTimeoutError(f"Reply API call failed: {exc}") from exc

            page = parse_continuation_response(raw, video_id, is_reply=True)

            for reply in page.comments:
                reply.parent_comment_id  = comment_id
                reply.scrape_job_id      = job_id
                accumulator.append(reply)

            if len(accumulator) >= SUB_BATCH_SIZE or page.is_last_page:
                if accumulator:
                    written, dupes = await _write_sub_batch(
                        accumulator, db, history_repo, job_id
                    )
                    result.comments_written  += written
                    result.duplicates_skipped += dupes
                    result.sub_batches_done   += 1
                    accumulator.clear()

            if page.is_last_page:
                result.is_exhausted = True
                break

            token = page.next_token
            # No delay for replies — each chain uses a unique token and is
            # far less likely to trigger rate limiting than repeated TLC calls.

    return result


# ── Sub-batch writer ───────────────────────────────────────────────────────

async def _write_sub_batch(
    comments:     list[CommentDocument],
    db:           AsyncIOMotorDatabase,
    history_repo: CommentHistoryRepository,
    job_id:       str,
) -> tuple[int, int]:
    """
    Write a list of CommentDocuments to MongoDB.
    Handles:
      1. New comments   → insert
      2. Existing, unchanged → update last_seen_at only
      3. Existing, edited  → archive old version, update with new text

    Returns (inserted_count, duplicate_count).
    """
    if not comments:
        return 0, 0

    col = db["comments"]

    # Split into new vs potentially existing
    comment_ids = [c.comment_id for c in comments]
    video_id    = comments[0].video_id

    # Fetch existing documents for edit detection (batch lookup)
    existing_raw = await col.find(
        {"video_id": video_id, "comment_id": {"$in": comment_ids}},
        {"comment_id": 1, "text_hash": 1, "version": 1, "like_count": 1},
    ).to_list(length=len(comments))

    existing_map = {d["comment_id"]: d for d in existing_raw}

    to_insert:  list[dict] = []
    update_ops: list       = []

    from pymongo import UpdateOne

    for comment in comments:
        existing = existing_map.get(comment.comment_id)

        if existing is None:
            # Brand new comment
            doc = comment.to_dict()
            to_insert.append(doc)
        else:
            # Comment we've seen before
            if comment.text_hash != existing.get("text_hash", ""):
                # ── EDIT DETECTED ──────────────────────────────────────────
                old_version = existing.get("version", 1)
                history     = CommentHistoryDocument(
                    comment_id              = comment.comment_id,
                    video_id                = video_id,
                    version                 = old_version,
                    text                    = "",   # we don't have the old text here
                    text_hash               = existing["text_hash"],
                    like_count_at_detection = existing.get("like_count", 0),
                    detected_by_job_id      = job_id,
                )
                await history_repo.archive_version(history)

                update_ops.append(UpdateOne(
                    {"video_id": video_id, "comment_id": comment.comment_id},
                    {"$set": {
                        "text":              comment.text,
                        "text_hash":         comment.text_hash,
                        "text_formatted":    comment.text_formatted,
                        "is_edited":         True,
                        "edit_detected_at":  datetime.now(timezone.utc),
                        "like_count":        comment.like_count,
                        "like_count_display":comment.like_count_display,
                        "reply_count":       comment.reply_count,
                        "is_hearted":        comment.is_hearted,
                        "last_seen_at":      datetime.now(timezone.utc),
                        "status":            CommentStatus.ACTIVE,
                    },
                     "$inc": {"version": 1}},
                ))
            else:
                # Unchanged — just refresh last_seen_at and engagement
                update_ops.append(UpdateOne(
                    {"video_id": video_id, "comment_id": comment.comment_id},
                    {"$set": {
                        "last_seen_at":      datetime.now(timezone.utc),
                        "like_count":        comment.like_count,
                        "like_count_display":comment.like_count_display,
                        "reply_count":       comment.reply_count,
                        "is_hearted":        comment.is_hearted,
                        "status":            CommentStatus.ACTIVE,
                    }},
                ))

    inserted   = 0
    duplicates = 0

    # ── Bulk insert new comments ────────────────────────────────────────
    if to_insert:
        try:
            res         = await col.insert_many(to_insert, ordered=False)
            inserted    = len(res.inserted_ids)
            duplicates  = len(to_insert) - inserted
        except Exception as exc:
            if hasattr(exc, "details"):
                inserted   = exc.details.get("nInserted", 0)
                duplicates = len(to_insert) - inserted
            else:
                raise DatabaseOperationError(
                    f"insert_many failed: {exc}", detail=str(exc)
                ) from exc

    # ── Bulk update existing comments ───────────────────────────────────
    if update_ops:
        await col.bulk_write(update_ops, ordered=False)

    logger.debug(
        "sub_batch_written",
        inserted=inserted,
        updated=len(update_ops),
        duplicates=duplicates,
        video_id=video_id,
    )
    return inserted, duplicates


# ── Checkpoint helper ──────────────────────────────────────────────────────

async def _checkpoint(
    *,
    job_id:                  str,
    batch_id:                str,
    batch_number:            int,
    token:                   str,
    result:                  BatchResult,
    cache:                   CacheManager,
    session_repo:            ScrapeSessionRepository,
    batch_repo:              ScrapeBatchRepository,
    context:                 InnertubeContext,
    write_session_checkpoint: bool = True,
) -> None:
    """
    Save progress to Redis (fast) and MongoDB (durable) after every sub-batch.
    Order: Redis first (speed), MongoDB second (durability).
    If MongoDB write fails, Redis still has it — next checkpoint will retry.

    write_session_checkpoint controls whether the MongoDB session doc is
    updated.  The session doc is used only for job resume; writing it on
    every sub-batch is wasteful.  Callers should set it True only on the
    first sub-batch of each Celery task (guaranteed durable checkpoint)
    and False for subsequent sub-batches within the same task.
    """
    now = datetime.now(timezone.utc)

    # ── Redis ──────────────────────────────────────────────────────────
    await cache.update_tlc_token(
        job_id           = job_id,
        token            = token,
        sub_batch_number = result.sub_batches_done,
        comments_written = result.comments_written,
        batch_number     = batch_number,
    )

    # ── MongoDB batch doc ──────────────────────────────────────────────
    await batch_repo.checkpoint(
        batch_id,
        sub_batches_done   = result.sub_batches_done,
        comments_written   = result.comments_written,
        duplicates_skipped = result.duplicates_skipped,
        reply_tokens_found = result.reply_tokens_found,
        current_token      = token,
    )

    # ── MongoDB session doc (only when requested) ──────────────────────
    if write_session_checkpoint:
        await session_repo.checkpoint(
            job_id                  = job_id,
            token                   = token,
            token_obtained_at       = context.token_obtained_at
                if hasattr(context, "token_obtained_at") else now,
            sub_batch_number        = result.sub_batches_done,
            comments_written_total  = result.comments_written,
            current_batch_number    = batch_number,
        )


# ── HTTP fetch with retry ──────────────────────────────────────────────────

async def _fetch_with_retry(scraper: ScraperSession, token: str) -> dict:
    """
    Call post_continuation() with retry logic.
    Handles: rate limits (429), server errors (5xx), network errors.
    """
    import aiohttp

    for attempt in range(3):
        try:
            return await scraper.post_continuation(token)

        except aiohttp.ClientResponseError as exc:
            if exc.status in PERMANENT_ERROR_CODES:
                raise ScraperVideoNotFoundError(
                    f"Video not accessible (HTTP {exc.status})"
                )
            if exc.status in RATE_LIMIT_STATUS_CODES:
                wait = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                logger.warning("rate_limited", attempt=attempt, wait_seconds=wait)
                await asyncio.sleep(wait)
                if attempt == 2:
                    raise ScraperRateLimitError(
                        f"Rate limited after {attempt + 1} attempts"
                    )
                continue
            if 500 <= exc.status < 600:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

        except asyncio.TimeoutError:
            logger.warning("request_timeout", attempt=attempt)
            await asyncio.sleep(5 * (attempt + 1))
            if attempt == 2:
                raise ScraperTimeoutError("Request timed out after 3 attempts")

        except Exception as exc:
            await asyncio.sleep(3 * (attempt + 1))
            if attempt == 2:
                raise

    raise ScraperTimeoutError("All retry attempts exhausted")
