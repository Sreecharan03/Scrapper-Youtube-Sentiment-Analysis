"""
app/workers/tasks/transcript_tasks.py
=======================================
Celery task that fetches a YouTube video transcript and stores it in MongoDB.

HOW IT WORKS:
  1. Mark transcript document as FETCHING in MongoDB
  2. Call youtube-transcript-api (sync — runs directly, no asyncio.run needed)
     to list all available caption tracks and fetch the best one
  3. If original language != "en" and translation is available → also fetch
     the YouTube-translated English version (for downstream NLP)
  4. Write all data to MongoDB via mark_completed()

MULTI-LANGUAGE FETCH STRATEGY:
  preferred_languages is a priority list, e.g. ["en", "hi"].
  The resolution order is:
    1. Manually uploaded transcript in a preferred language
    2. Auto-generated transcript in a preferred language
    3. Any transcript (first available — handles non-English videos
       where the caller didn't know the source language)
  This means fetch_transcript("es_video", preferred_languages=["en"]) will
  find the Spanish auto-captions and also store the YouTube→English translation.

WHY SYNC (not asyncio.run):
  youtube-transcript-api is a synchronous library (uses requests, not aiohttp).
  The DB writes are still async — wrapped in asyncio.run() at the end.
  Celery's prefork model is fine with this; each worker process has its own loop.

ERROR TYPES:
  TranscriptsDisabled  → mark UNAVAILABLE (permanent, no retry)
  NoTranscriptFound    → mark UNAVAILABLE (permanent, no retry)
  Exception (network)  → retry up to max_retries, then mark FAILED
"""

import asyncio
from datetime import datetime, timezone

from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
)

# Permanent errors — retrying will never help
_PERMANENT_ERRORS = (
    VideoUnavailable,     # video deleted, private, or doesn't exist
    VideoUnplayable,      # video exists but can't be played (geo-block, etc.)
    TranscriptsDisabled,  # owner disabled captions
    NoTranscriptFound,    # no caption track in requested language(s)
    AgeRestricted,        # age-gated — needs cookies
    InvalidVideoId,       # bad video ID format
)

# Retryable errors — network or YouTube rate-limiting
_RETRYABLE_ERRORS = (
    RequestBlocked,       # YouTube blocked our IP (temporary)
    IpBlocked,            # IP-level block
    YouTubeRequestFailed, # network error
)

from app.core.logging import get_logger
from app.scraper.pipeline import make_db_client
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


# ── Task ──────────────────────────────────────────────────────────────────

@celery_app.task(
    bind                = True,
    name                = "fetch_transcript",
    queue               = "scraper",
    max_retries         = 3,
    default_retry_delay = 60,
    soft_time_limit     = 120,   # transcript fetch is fast — 2 min is generous
    time_limit          = 150,
    acks_late           = True,
    ignore_result       = True,
)
def fetch_transcript(
    self,
    *,
    video_id:            str,
    preferred_languages: list = None,  # e.g. ["en", "hi", "es"] — order matters
) -> dict:
    """
    Fetch and store the transcript for `video_id`.

    Args:
        video_id:            YouTube video ID (11 chars)
        preferred_languages: Priority list of ISO 639-1 language codes.
                             Defaults to ["en"] — tries English first, falls
                             back to any available language.
    """
    if preferred_languages is None:
        preferred_languages = ["en"]

    logger.info(
        "transcript_fetch_started",
        video_id            = video_id,
        preferred_languages = preferred_languages,
        task_id             = self.request.id,
    )

    # ── Mark FETCHING in MongoDB ──────────────────────────────────────────
    asyncio.run(_mark_fetching(video_id))

    try:
        # ── Fetch from YouTube (sync) ─────────────────────────────────────
        result = _fetch_transcript_sync(video_id, preferred_languages)

        # ── Store in MongoDB (async) ──────────────────────────────────────
        asyncio.run(_store_completed(video_id, result))

        logger.info(
            "transcript_fetch_completed",
            video_id      = video_id,
            language      = result["original_language_code"],
            segments      = result["segment_count"],
            is_translated = result["is_translated"],
            duration_secs = result["total_duration_secs"],
        )
        return {"video_id": video_id, "status": "completed",
                "language": result["original_language_code"]}

    except _PERMANENT_ERRORS as exc:
        # Permanent — retrying will never help (no captions, video gone, age-gated, etc.)
        reason = str(exc) or type(exc).__name__
        logger.warning("transcript_unavailable", video_id=video_id,
                       reason=reason, error_type=type(exc).__name__)
        asyncio.run(_mark_unavailable(video_id, reason))
        return {"video_id": video_id, "status": "unavailable"}

    except Exception as exc:
        if self.request.retries < self.max_retries:
            backoff = 60 * (2 ** self.request.retries)   # 60s, 120s, 240s
            logger.warning(
                "transcript_fetch_retrying",
                video_id = video_id,
                attempt  = self.request.retries + 1,
                backoff  = backoff,
                error    = str(exc),
            )
            raise self.retry(exc=exc, countdown=backoff)

        # All retries exhausted
        logger.error("transcript_fetch_failed", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}


# ── Sync fetch logic ──────────────────────────────────────────────────────

def _fetch_transcript_sync(video_id: str, preferred_languages: list) -> dict:
    """
    Fetch the best available transcript using youtube-transcript-api.

    Returns a dict ready to pass to TranscriptRepository.mark_completed().

    RESOLUTION ORDER:
      1. Manually uploaded transcript in a preferred language
      2. Auto-generated transcript in a preferred language
      3. ANY transcript (first available — for videos where source language
         is unknown or not in preferred_languages list)

    ENGLISH TRANSLATION:
      If the best transcript is not English AND the track supports translation,
      also fetch the YouTube→English machine translation.  This is stored
      separately as english_segments and used by all downstream NLP.
    """
    api             = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    # Build available_languages list BEFORE selecting one (list() consumes iterator)
    available_languages = [
        {
            "language_code": t.language_code,
            "language_name": t.language,
            "is_generated":  t.is_generated,
        }
        for t in transcript_list
    ]

    # Re-list (iterator was consumed above)
    transcript_list = api.list(video_id)

    # ── Step 1: Find the best transcript ─────────────────────────────────
    transcript = None

    try:
        transcript = transcript_list.find_manually_created_transcript(preferred_languages)
    except NoTranscriptFound:
        pass

    if transcript is None:
        try:
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_generated_transcript(preferred_languages)
        except NoTranscriptFound:
            pass

    if transcript is None:
        # Fall back to whatever is available (e.g. a Korean video when caller asked for "en")
        transcript_list = api.list(video_id)
        transcript = next(iter(transcript_list))   # raises StopIteration → NoTranscriptFound if empty

    # ── Step 2: Fetch original segments ──────────────────────────────────
    fetched = transcript.fetch()
    original_segments = [
        {
            "start_ms": int(snippet.start * 1000),
            "end_ms":   int((snippet.start + snippet.duration) * 1000),
            "text":     snippet.text,
        }
        for snippet in fetched.snippets
    ]

    # ── Step 3: English translation (only if original is not English) ─────
    english_segments = None
    is_translated    = False

    if fetched.language_code != "en" and transcript.is_translatable:
        try:
            eng_fetched = transcript.translate("en").fetch()
            english_segments = [
                {
                    "start_ms": int(snippet.start * 1000),
                    "end_ms":   int((snippet.start + snippet.duration) * 1000),
                    "text":     snippet.text,
                }
                for snippet in eng_fetched.snippets
            ]
            is_translated = True
            logger.debug(
                "transcript_translated",
                video_id          = video_id,
                original_language = fetched.language_code,
                translated_to     = "en",
                segments          = len(english_segments),
            )
        except Exception as exc:
            # Translation is best-effort — don't fail the whole task
            logger.warning(
                "transcript_translation_failed",
                video_id = video_id,
                error    = str(exc),
            )

    # ── Step 4: Compute duration from last segment ────────────────────────
    total_duration_secs = 0.0
    if fetched.snippets:
        last = fetched.snippets[-1]
        total_duration_secs = round(last.start + last.duration, 2)

    return {
        "original_language_code": fetched.language_code,
        "original_language_name": fetched.language,
        "is_auto_generated":      fetched.is_generated,
        "available_languages":    available_languages,
        "original_segments":      original_segments,
        "english_segments":       english_segments,
        "is_translated":          is_translated,
        "segment_count":          len(original_segments),
        "total_duration_secs":    total_duration_secs,
    }


# ── Async DB helpers ──────────────────────────────────────────────────────

async def _mark_fetching(video_id: str) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.transcript_repo import TranscriptRepository
        await TranscriptRepository(db).mark_fetching(video_id)
    finally:
        mongo_client.close()


async def _store_completed(video_id: str, data: dict) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.transcript_repo import TranscriptRepository
        await TranscriptRepository(db).mark_completed(video_id, data)
    finally:
        mongo_client.close()


async def _mark_unavailable(video_id: str, reason: str) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.transcript_repo import TranscriptRepository
        await TranscriptRepository(db).mark_unavailable(video_id, reason)
    finally:
        mongo_client.close()


async def _mark_failed(video_id: str, error: str) -> None:
    mongo_client, db = make_db_client()
    try:
        from app.db.repositories.transcript_repo import TranscriptRepository
        await TranscriptRepository(db).mark_failed(video_id, error)
    finally:
        mongo_client.close()
