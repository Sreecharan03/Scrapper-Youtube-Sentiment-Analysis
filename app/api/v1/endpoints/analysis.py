"""
app/api/v1/endpoints/analysis.py
==================================
FastAPI endpoints for comment classification and audience intelligence.

ENDPOINTS:
  POST /api/v1/analysis/{video_id}/classify  → dispatch classify_comments Celery task (202)
  GET  /api/v1/analysis/{video_id}           → return comment_analysis aggregate doc

PREREQUISITE: summary must be completed before classification can run.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.analysis import ClassifyCommentsResponse, CommentAnalysisResponse
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.classification_repo import ClassificationRepository, ClassificationStatus
from app.db.repositories.summary_repo import SummaryRepository
from app.models.summary import SummaryStatus

router = APIRouter(prefix="/analysis", tags=["Analysis"])
logger = get_logger(__name__)


def get_classification_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> ClassificationRepository:
    return ClassificationRepository(db)


def get_summary_repo(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> SummaryRepository:
    return SummaryRepository(db)


@router.post(
    "/{video_id}/classify",
    response_model = ClassifyCommentsResponse,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Classify all comments for a video",
    description    = (
        "Dispatch a Fireworks AI task to classify all comments into intent labels "
        "(question/praise/criticism/confusion/misconception/request/spam) and sentiment. "
        "Summary must be generated first via POST /api/v1/summaries/{video_id}.\n\n"
        "Returns 409 if classification is already in progress.\n"
        "Returns 422 if summary is not yet completed."
    ),
)
async def classify_comments(
    video_id:            str,
    classification_repo: ClassificationRepository = Depends(get_classification_repo),
    summary_repo:        SummaryRepository        = Depends(get_summary_repo),
) -> ClassifyCommentsResponse:

    # Guard: summary must be completed first
    summary_status = await summary_repo.get_status(video_id)
    if summary_status != SummaryStatus.COMPLETED:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                f"Summary for {video_id!r} is not ready (status={summary_status!r}). "
                "Generate the summary first via POST /api/v1/summaries/{video_id}."
            ),
        )

    # Guard: don't fire duplicate while one is running
    current = await classification_repo.get_status(video_id)
    if current == ClassificationStatus.PROCESSING:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = f"Classification for {video_id!r} is already in progress.",
        )

    from app.workers.tasks.classification_tasks import classify_comments as _task
    _task.apply_async(kwargs={"video_id": video_id}, queue="scraper")

    logger.info("classification_dispatched", video_id=video_id)

    return ClassifyCommentsResponse(
        video_id = video_id,
        status   = "pending",
        message  = (
            f"Classification queued for {video_id!r}. "
            f"Poll GET /api/v1/analysis/{video_id} for status and results."
        ),
    )


@router.post(
    "/{video_id}/classify/retry",
    response_model = ClassifyCommentsResponse,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Retry classification for failed comments only",
    description    = "Re-run classification only on comments that previously failed. "
                     "Returns 404 if no failed comments exist.",
)
async def retry_failed_classification(
    video_id:            str,
    classification_repo: ClassificationRepository = Depends(get_classification_repo),
) -> ClassifyCommentsResponse:

    current = await classification_repo.get_status(video_id)
    if current == ClassificationStatus.PROCESSING:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = f"Classification for {video_id!r} is already in progress.",
        )

    from app.workers.tasks.classification_tasks import classify_comments as _task
    _task.apply_async(kwargs={"video_id": video_id, "retry_failed": True}, queue="scraper")

    logger.info("classification_retry_dispatched", video_id=video_id)

    return ClassifyCommentsResponse(
        video_id = video_id,
        status   = "pending",
        message  = (
            f"Retry queued for failed comments in {video_id!r}. "
            f"Poll GET /api/v1/analysis/{video_id} for status and results."
        ),
    )


@router.get(
    "/{video_id}",
    response_model = CommentAnalysisResponse,
    summary        = "Get comment analysis results for a video",
)
async def get_analysis(
    video_id:            str,
    classification_repo: ClassificationRepository = Depends(get_classification_repo),
) -> CommentAnalysisResponse:

    doc = await classification_repo.get_analysis(video_id)
    if not doc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = (
                f"No analysis found for {video_id!r}. "
                f"Submit POST /api/v1/analysis/{video_id}/classify first."
            ),
        )
    return CommentAnalysisResponse.from_document(doc)
