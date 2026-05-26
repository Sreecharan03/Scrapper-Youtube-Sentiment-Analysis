"""
app/api/v1/endpoints/transcripts.py
=====================================
FastAPI route handlers for transcript fetch and retrieval.

ENDPOINTS:
  POST /api/v1/transcripts/{video_id}
    → Dispatch a Celery task to fetch the transcript.
    → Accepts optional preferred_languages in request body.
    → Returns 202 immediately. Poll GET to check progress.
    → Returns 409 if a fetch is already in progress.

  GET /api/v1/transcripts/{video_id}
    → Return full transcript including all segments.
    → Use ?segments=false to get status only (no segment arrays).

LANGUAGE PARAMETER:
  preferred_languages defaults to ["en"].
  For a non-English video, pass the expected language first:
    POST body: {"preferred_languages": ["ko", "en"]}
  If the video has Korean captions, those are fetched + English translation stored.
  If only auto-generated English exists, that is used.
"""

from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.transcript import (
    FetchTranscriptRequest,
    FetchTranscriptResponse,
    TranscriptResponse,
    TranscriptStatusResponse,
)
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.transcript_repo import TranscriptRepository
from app.models.transcript import TranscriptStatus

router = APIRouter(prefix="/transcripts", tags=["Transcripts"])
logger = get_logger(__name__)


def get_transcript_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> TranscriptRepository:
    return TranscriptRepository(db)


# ── POST /transcripts/{video_id} ──────────────────────────────────────────

@router.post(
    "/{video_id}",
    response_model    = FetchTranscriptResponse,
    status_code       = status.HTTP_202_ACCEPTED,
    summary           = "Fetch transcript for a video",
    description       = (
        "Dispatch a background task to fetch the YouTube transcript for `video_id`.\n\n"
        "**Multi-language support:** Pass `preferred_languages` as a priority list. "
        "The fetcher tries each language in order. If none match, it falls back to "
        "any available transcript. For non-English videos, an English translation "
        "is also stored automatically when YouTube supports it.\n\n"
        "**Idempotency:** If a fetch is already in progress for this video, returns 409. "
        "If a completed transcript already exists, this re-fetches and overwrites it."
    ),
)
async def fetch_transcript(
    video_id: str,
    request:  FetchTranscriptRequest = Body(default_factory=FetchTranscriptRequest),
    repo:     TranscriptRepository   = Depends(get_transcript_repo),
) -> FetchTranscriptResponse:

    # Guard: don't fire duplicate fetch while one is already running
    current_status = await repo.get_status(video_id)
    if current_status == TranscriptStatus.FETCHING:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = f"Transcript fetch for video {video_id!r} is already in progress.",
        )

    # Dispatch Celery task
    from app.workers.tasks.transcript_tasks import fetch_transcript as _task
    _task.apply_async(
        kwargs = {
            "video_id":            video_id,
            "preferred_languages": request.preferred_languages,
        },
        queue = "scraper",
    )

    logger.info(
        "transcript_fetch_dispatched",
        video_id            = video_id,
        preferred_languages = request.preferred_languages,
    )

    return FetchTranscriptResponse(
        video_id            = video_id,
        status              = "pending",
        preferred_languages = request.preferred_languages,
        message             = (
            f"Transcript fetch queued for video {video_id!r}. "
            f"Preferred languages: {request.preferred_languages}. "
            f"Poll GET /api/v1/transcripts/{video_id} for status."
        ),
    )


# ── GET /transcripts/{video_id} ───────────────────────────────────────────

@router.get(
    "/{video_id}",
    response_model = TranscriptResponse,
    summary        = "Get transcript for a video",
    description    = (
        "Return the stored transcript for `video_id`.\n\n"
        "Use `?segments=false` to get status and language metadata only "
        "(no segment arrays — faster for polling).\n\n"
        "When `status` is `completed`:\n"
        "- `original_segments`: caption text in the original language\n"
        "- `english_segments`: YouTube-translated English (null if original is English)\n\n"
        "When `status` is `pending` or `fetching`, segment arrays are empty."
    ),
)
async def get_transcript(
    video_id: str,
    segments: bool = Query(True, description="Include segment arrays in response"),
    repo:     TranscriptRepository = Depends(get_transcript_repo),
) -> TranscriptResponse:

    doc = await repo.get_transcript(video_id)
    if not doc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                f"No transcript found for video {video_id!r}. "
                f"Submit a POST /api/v1/transcripts/{video_id} request first."
            ),
        )
    return TranscriptResponse.from_document(doc, include_segments=segments)
