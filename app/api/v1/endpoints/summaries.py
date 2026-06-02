"""
app/api/v1/endpoints/summaries.py
====================================
FastAPI endpoints for LLM summary generation and retrieval.

ENDPOINTS:
  POST /api/v1/summaries/{video_id}  → dispatch generate_summary Celery task (202)
  GET  /api/v1/summaries/{video_id}  → return stored summary

PREREQUISITE: transcript must be completed before summary can be generated.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.summary import GenerateSummaryResponse, SummaryResponse
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.summary_repo import SummaryRepository
from app.db.repositories.transcript_repo import TranscriptRepository
from app.models.summary import SummaryStatus
from app.models.transcript import TranscriptStatus

router = APIRouter(prefix="/summaries", tags=["Summaries"])
logger = get_logger(__name__)


def get_summary_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> SummaryRepository:
    return SummaryRepository(db)


def get_transcript_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> TranscriptRepository:
    return TranscriptRepository(db)


@router.post(
    "/{video_id}",
    response_model = GenerateSummaryResponse,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Generate LLM summary for a video",
    description    = (
        "Dispatch a Claude Haiku task to generate a structured summary from the "
        "video transcript. Transcript must be fetched first via "
        "POST /api/v1/transcripts/{video_id}.\n\n"
        "Returns 409 if generation is already in progress.\n"
        "Returns 422 if transcript is not yet completed."
    ),
)
async def generate_summary(
    video_id:        str,
    summary_repo:    SummaryRepository    = Depends(get_summary_repo),
    transcript_repo: TranscriptRepository = Depends(get_transcript_repo),
) -> GenerateSummaryResponse:

    # Guard: don't fire duplicate while one is running
    current = await summary_repo.get_status(video_id)
    if current == SummaryStatus.GENERATING:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = f"Summary generation for {video_id!r} is already in progress.",
        )

    # Guard: transcript must be completed first
    transcript_status = await transcript_repo.get_status(video_id)
    if transcript_status != TranscriptStatus.COMPLETED:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                f"Transcript for {video_id!r} is not ready "
                f"(status={transcript_status!r}). "
                "Fetch the transcript first via POST /api/v1/transcripts/{video_id}."
            ),
        )

    from app.workers.tasks.summary_tasks import generate_summary as _task
    _task.apply_async(kwargs={"video_id": video_id}, queue="scraper")

    logger.info("summary_generation_dispatched", video_id=video_id)

    return GenerateSummaryResponse(
        video_id = video_id,
        status   = "pending",
        message  = (
            f"Summary generation queued for {video_id!r}. "
            f"Poll GET /api/v1/summaries/{video_id} for status."
        ),
    )


@router.get(
    "/{video_id}",
    response_model = SummaryResponse,
    summary        = "Get LLM summary for a video",
)
async def get_summary(
    video_id:     str,
    summary_repo: SummaryRepository = Depends(get_summary_repo),
) -> SummaryResponse:

    doc = await summary_repo.get_summary(video_id)
    if not doc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                f"No summary found for {video_id!r}. "
                f"Submit POST /api/v1/summaries/{video_id} first."
            ),
        )
    return SummaryResponse.from_document(doc)
