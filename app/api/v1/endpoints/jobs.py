"""
app/api/v1/endpoints/jobs.py
=============================
FastAPI route handlers for scrape job management.

ENDPOINTS:
  POST   /api/v1/jobs                    → Submit a new scrape job
  GET    /api/v1/jobs                    → List all jobs (paginated)
  GET    /api/v1/jobs/{job_id}           → Get a specific job's status & progress
  GET    /api/v1/jobs/{job_id}/batches   → List all batch documents for a job
  POST   /api/v1/jobs/{job_id}/resume    → Resume a paused job

DESIGN NOTES:
  - Route handlers are thin — they validate input, call the repository,
    dispatch a Celery task, and return a response. No business logic here.
  - Database is injected via FastAPI's Depends() — makes testing trivial.
  - All handlers are async — never block the event loop.
  - Job creation now dispatches scrape_job_start (Phase 2) instead of the
    Phase 1 stub task.
"""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.schemas.job import (
    BatchListResponse,
    BatchSummary,
    CreateJobRequest,
    JobListResponse,
    JobResponse,
    ResumeJobResponse,
)
from app.core.cache import cache
from app.core.logging import get_logger
from app.db.connection import get_database
from app.db.repositories.job_repo import JobRepository
from app.db.repositories.scrape_batch_repo import ScrapeBatchRepository
from app.models.job import JobDocument, JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])
logger = get_logger(__name__)


# ── Dependencies ──────────────────────────────────────────────────────────

def get_job_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> JobRepository:
    return JobRepository(db)

def get_batch_repo(db: AsyncIOMotorDatabase = Depends(get_database)) -> ScrapeBatchRepository:
    return ScrapeBatchRepository(db)


# ── Utility ───────────────────────────────────────────────────────────────

_YT_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from a URL."""
    match = _YT_VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a scrape job",
    description=(
        "Queue a background job to scrape all comments for a YouTube video. "
        "Returns immediately with job details. Poll GET /jobs/{job_id} for progress."
    ),
)
async def create_job(
    request: CreateJobRequest,
    repo: JobRepository = Depends(get_job_repo),
) -> JobResponse:
    video_id = extract_video_id(request.video_url)
    if not video_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract a valid YouTube video ID from the URL.",
        )

    # Prevent duplicate active jobs for the same video
    if await repo.has_active_job(video_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A job for video {video_id!r} is already queued or running.",
        )

    # Acquire the distributed job lock (prevents race condition if two API
    # pods process the same request simultaneously)
    if not await cache.acquire_job_lock(video_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A job for video {video_id!r} is already being initialized.",
        )

    # Create the job document in MongoDB
    job    = JobDocument(video_id=video_id, video_url=request.video_url)
    job_id = await repo.create_job(job)

    # Dispatch scrape_job_start (non-blocking — returns immediately)
    # Import here to avoid circular import at module load time
    try:
        from app.workers.tasks.job_tasks import scrape_job_start
        scrape_job_start.apply_async(
            kwargs={"job_id": job_id, "video_id": video_id},
            queue="scraper",
        )
        logger.info("job_created", job_id=job_id, video_id=video_id)
    except Exception as exc:
        # If Celery broker is unreachable the job doc already exists in MongoDB.
        # Return 202 anyway — the job can be retried/resumed once the worker is up.
        logger.error(
            "job_dispatch_failed",
            job_id   = job_id,
            video_id = video_id,
            error    = str(exc),
        )

    doc = await repo.get_job(job_id)
    return JobResponse.from_document(doc)


@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
)
async def list_jobs(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status. Valid values: pending | fetching_meta | "
                    "scraping_tlcs | finalizing | completed | paused_batch_failed | "
                    "paused_token_expired | failed_permanent",
    ),
    skip:  int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Page size"),
    repo: JobRepository = Depends(get_job_repo),
) -> JobListResponse:
    jobs  = await repo.list_jobs(status=status_filter, skip=skip, limit=limit)
    total = await repo.count({"status": status_filter} if status_filter else {})
    return JobListResponse(
        jobs  = [JobResponse.from_document(j) for j in jobs],
        total = total,
        skip  = skip,
        limit = limit,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status and progress",
)
async def get_job(
    job_id: str,
    repo: JobRepository = Depends(get_job_repo),
) -> JobResponse:
    doc = await repo.get_job(job_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )
    return JobResponse.from_document(doc)


@router.get(
    "/{job_id}/batches",
    response_model=BatchListResponse,
    summary="List all scrape batches for a job",
    description=(
        "Returns one BatchSummary per 5 000-comment batch. "
        "Useful for diagnosing which batch stalled or failed."
    ),
)
async def list_batches(
    job_id: str,
    job_repo:   JobRepository        = Depends(get_job_repo),
    batch_repo: ScrapeBatchRepository = Depends(get_batch_repo),
) -> BatchListResponse:
    # Validate job exists
    job = await job_repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )

    batches = await batch_repo.list_batches_for_job(job_id)
    return BatchListResponse(
        job_id  = job_id,
        batches = [BatchSummary.from_document(b) for b in batches],
        total   = len(batches),
    )


@router.post(
    "/{job_id}/resume",
    response_model=ResumeJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume a paused job",
    description=(
        "Restart a job that was paused due to a batch failure or token expiry. "
        "Re-reads the last saved continuation token from the scrape_session "
        "and re-queues the next TLC batch."
    ),
)
async def resume_job(
    job_id: str,
    job_repo:   JobRepository        = Depends(get_job_repo),
    batch_repo: ScrapeBatchRepository = Depends(get_batch_repo),
    db: AsyncIOMotorDatabase          = Depends(get_database),
) -> ResumeJobResponse:
    job = await job_repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )

    current_status = job.get("status")
    if current_status not in JobStatus.PAUSED_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id!r} is in status {current_status!r} — "
                f"only paused jobs can be resumed. "
                f"Paused statuses: {sorted(JobStatus.PAUSED_STATUSES)}"
            ),
        )

    # Re-read the last saved token from MongoDB scrape_session
    from app.db.repositories.scrape_session_repo import ScrapeSessionRepository
    session_repo = ScrapeSessionRepository(db)
    session      = await session_repo.get_session(job_id)

    if not session or not session.get("current_tlc_token"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No saved continuation token found for job {job_id!r}. "
                "Cannot resume — the job must be restarted from scratch."
            ),
        )

    video_id     = job["video_id"]
    batch_number = session.get("current_batch_number", 1)
    token        = session["current_tlc_token"]

    # Transition job back to SCRAPING_TLCS
    await job_repo.resume_job(job_id)

    # Create a new batch document for the resumed batch
    from app.models.scrape_batch import ScrapeBatchDocument
    new_batch = ScrapeBatchDocument(
        job_id         = job_id,
        batch_number   = batch_number,
        token_at_start = token,
    )
    batch_id = await batch_repo.create_batch(new_batch)

    # We need the InnertubeContext to re-start the scrape.
    # For a resumed job, we re-fetch the video page to get a fresh context.
    # This is done inside scrape_job_start if context is missing, OR we
    # fire scrape_tlc_batch with a "needs_context_refresh=True" flag.
    # Simplest approach: fire scrape_job_start in resume mode.
    from app.workers.tasks.job_tasks import scrape_job_start
    scrape_job_start.apply_async(
        kwargs={"job_id": job_id, "video_id": video_id},
        queue="scraper",
    )

    logger.info("job_resumed", job_id=job_id, video_id=video_id,
                batch_number=batch_number)

    return ResumeJobResponse(
        job_id  = job_id,
        status  = JobStatus.SCRAPING_TLCS,
        message = (
            f"Job resumed from batch {batch_number}. "
            f"Re-fetching YouTube context before continuing."
        ),
    )
