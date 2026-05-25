"""
app/api/v1/schemas/job.py
==========================
Pydantic models for Job API request/response contracts.

WHY SEPARATE FROM app/models/job.py:
  - app/models/job.py  = MongoDB document shape (what Atlas stores)
  - app/api/schemas/   = HTTP contract (what callers send and receive)
  They're intentionally different:
    - The DB document has retry_count, celery_task_id, raw error_type.
    - The API response exposes clean progress/reply fields instead.
    - Internal fields (celery_task_id) are never exposed to callers.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Request models ────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    """Request body for POST /api/v1/jobs"""

    video_url: str = Field(
        ...,
        description="Full YouTube video URL",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )

    @field_validator("video_url")
    @classmethod
    def must_be_youtube_url(cls, v: str) -> str:
        """Reject non-YouTube URLs before they reach the scraper."""
        v = v.strip()
        if "youtube.com/watch" not in v and "youtu.be/" not in v:
            raise ValueError(
                "URL must be a YouTube watch URL "
                "(youtube.com/watch?v=... or youtu.be/...)"
            )
        return v


# ── Response models ───────────────────────────────────────────────────────

class BatchSummary(BaseModel):
    """Embedded summary for one scrape batch — used in GET /jobs/{id}/batches."""
    batch_id:           str
    batch_number:       int
    status:             str
    comments_written:   int           = 0
    duplicates_skipped: int           = 0
    reply_tokens_found: int           = 0
    sub_batches_done:   int           = 0
    token_at_start:     Optional[str] = None
    token_at_end:       Optional[str] = None
    created_at:         Optional[datetime] = None
    completed_at:       Optional[datetime] = None

    @classmethod
    def from_document(cls, doc: dict) -> "BatchSummary":
        return cls(
            batch_id           = str(doc["_id"]),
            batch_number       = doc["batch_number"],
            status             = doc["status"],
            comments_written   = doc.get("comments_written", 0),
            duplicates_skipped = doc.get("duplicates_skipped", 0),
            reply_tokens_found = doc.get("reply_tokens_found", 0),
            sub_batches_done   = doc.get("sub_batches_done", 0),
            token_at_start     = doc.get("token_at_start"),
            token_at_end       = doc.get("token_at_end"),
            created_at         = doc.get("created_at"),
            completed_at       = doc.get("completed_at"),
        )


class JobResponse(BaseModel):
    """Response body for job endpoints."""

    job_id:   str = Field(..., description="MongoDB document ID for this job")
    video_id: str
    video_url: str
    status:   str

    # Phase / sub-state description
    phase: Optional[str] = None

    # Overall progress
    comments_collected:      int           = 0
    total_comments_expected: Optional[int] = None
    progress_pct:            Optional[float] = Field(
        None, description="0–100 progress percentage (None if total unknown)"
    )

    # Batch-chain tracking
    current_batch_number:    int  = 0
    total_batches_completed: int  = 0
    tlc_completed:           bool = False

    # Reply pool tracking
    reply_tokens_found:     int           = 0
    reply_tokens_completed: int           = 0
    reply_tokens_failed:    int           = 0
    reply_completion_pct:   Optional[float] = None

    # Error info (only present in PAUSED / FAILED states)
    error_message: Optional[str] = None

    # Timestamps
    created_at:   datetime
    started_at:   Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at:   Optional[datetime] = None

    @classmethod
    def from_document(cls, doc: dict) -> "JobResponse":
        """Convert a raw MongoDB document dict to a clean API response."""
        total     = doc.get("total_comments_expected")
        collected = doc.get("comments_collected", 0)
        progress  = (
            round((collected / total) * 100, 1)
            if total and total > 0 else None
        )

        tokens_found = doc.get("reply_tokens_found", 0)
        reply_pct    = None
        if tokens_found > 0:
            done      = (doc.get("reply_tokens_completed", 0) +
                         doc.get("reply_tokens_failed", 0))
            reply_pct = round((done / tokens_found) * 100, 1)

        return cls(
            job_id                  = str(doc["_id"]),
            video_id                = doc["video_id"],
            video_url               = doc["video_url"],
            status                  = doc["status"],
            phase                   = doc.get("phase"),
            comments_collected      = collected,
            total_comments_expected = total,
            progress_pct            = progress,
            current_batch_number    = doc.get("current_batch_number", 0),
            total_batches_completed = doc.get("total_batches_completed", 0),
            tlc_completed           = doc.get("tlc_completed", False),
            reply_tokens_found      = tokens_found,
            reply_tokens_completed  = doc.get("reply_tokens_completed", 0),
            reply_tokens_failed     = doc.get("reply_tokens_failed", 0),
            reply_completion_pct    = reply_pct,
            error_message           = doc.get("error_message"),
            created_at              = doc["created_at"],
            started_at              = doc.get("started_at"),
            completed_at            = doc.get("completed_at"),
            updated_at              = doc.get("updated_at"),
        )


class JobListResponse(BaseModel):
    """Paginated list of jobs."""
    jobs:  list[JobResponse]
    total: int
    skip:  int
    limit: int


class BatchListResponse(BaseModel):
    """Batch detail list for a single job."""
    job_id:  str
    batches: list[BatchSummary]
    total:   int


class ResumeJobResponse(BaseModel):
    """Response for POST /api/v1/jobs/{job_id}/resume"""
    job_id:  str
    status:  str
    message: str
