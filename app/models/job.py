"""
app/models/job.py
==================
MongoDB document model for a scrape job.

COLLECTION: jobs

JOB LIFECYCLE (state machine):
  PENDING
    → FETCHING_META   (worker picked up the job, fetching video page)
    → SCRAPING_TLCS   (TLC batch chain running + reply pool running in parallel)
    → FINALIZING      (all TLC batches done + all replies drained)
    → COMPLETED       ✓ terminal

  Any active state can transition to:
    → PAUSED_BATCH_FAILED   (a batch exhausted retries — operator must resume)
    → PAUSED_TOKEN_EXPIRED  (continuation token expired mid-scrape)
    → FAILED_PERMANENT      (video gone private/deleted — no recovery possible)

  PAUSED states can transition back to:
    → SCRAPING_TLCS   (after operator calls resume API)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class JobStatus:
    # Active states
    PENDING               = "pending"
    FETCHING_META         = "fetching_meta"
    SCRAPING_TLCS         = "scraping_tlcs"
    FINALIZING            = "finalizing"
    # Terminal — success
    COMPLETED             = "completed"
    # Paused — recoverable
    PAUSED_BATCH_FAILED   = "paused_batch_failed"
    PAUSED_TOKEN_EXPIRED  = "paused_token_expired"
    # Terminal — unrecoverable
    FAILED_PERMANENT      = "failed_permanent"

    ACTIVE_STATUSES = {PENDING, FETCHING_META, SCRAPING_TLCS, FINALIZING}
    PAUSED_STATUSES = {PAUSED_BATCH_FAILED, PAUSED_TOKEN_EXPIRED}
    TERMINAL_STATUSES = {COMPLETED, FAILED_PERMANENT}


@dataclass
class JobDocument:
    # ── Identity ──────────────────────────────────────────────────────────
    video_id:  str
    video_url: str

    # ── Status ────────────────────────────────────────────────────────────
    status: str = JobStatus.PENDING
    phase:  str = "pending"   # human-readable sub-phase label

    # ── TLC batch chain tracking ──────────────────────────────────────────
    current_batch_number:   int  = 0
    total_batches_completed:int  = 0
    tlc_completed:          bool = False   # True when last TLC batch fires

    # ── Reply pool tracking ───────────────────────────────────────────────
    reply_tokens_found:     int = 0   # total tokens enqueued
    reply_tokens_completed: int = 0   # tokens successfully drained
    reply_tokens_failed:    int = 0   # tokens moved to failed_replies

    # ── Overall progress ──────────────────────────────────────────────────
    comments_collected:       int           = 0
    total_comments_expected:  Optional[int] = None   # from video metadata (approx)
    celery_task_id:           Optional[str] = None   # coordinator task ID

    # ── Error info (for PAUSED / FAILED states) ───────────────────────────
    error_message:  Optional[str] = None
    error_type:     Optional[str] = None
    retry_count:    int           = 0

    # ── Scrape config (immutable after job starts) ─────────────────────────
    batch_size:    int = 5000   # TLCs per Celery task
    sub_batch_size:int = 100    # writes per DB flush within a task

    # ── Timing ────────────────────────────────────────────────────────────
    created_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at:   Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── MongoDB _id ───────────────────────────────────────────────────────
    _id: Optional[str] = None

    # ── Computed helpers ──────────────────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return self.status in JobStatus.ACTIVE_STATUSES

    @property
    def is_paused(self) -> bool:
        return self.status in JobStatus.PAUSED_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in JobStatus.TERMINAL_STATUSES

    @property
    def progress_pct(self) -> Optional[float]:
        if self.total_comments_expected and self.total_comments_expected > 0:
            return round((self.comments_collected / self.total_comments_expected) * 100, 1)
        return None

    @property
    def reply_completion_pct(self) -> Optional[float]:
        if self.reply_tokens_found > 0:
            done = self.reply_tokens_completed + self.reply_tokens_failed
            return round((done / self.reply_tokens_found) * 100, 1)
        return None

    def to_dict(self) -> dict:
        return {
            "video_id":                self.video_id,
            "video_url":               self.video_url,
            "status":                  self.status,
            "phase":                   self.phase,
            "current_batch_number":    self.current_batch_number,
            "total_batches_completed": self.total_batches_completed,
            "tlc_completed":           self.tlc_completed,
            "reply_tokens_found":      self.reply_tokens_found,
            "reply_tokens_completed":  self.reply_tokens_completed,
            "reply_tokens_failed":     self.reply_tokens_failed,
            "comments_collected":      self.comments_collected,
            "total_comments_expected": self.total_comments_expected,
            "celery_task_id":          self.celery_task_id,
            "error_message":           self.error_message,
            "error_type":              self.error_type,
            "retry_count":             self.retry_count,
            "batch_size":              self.batch_size,
            "sub_batch_size":          self.sub_batch_size,
            "created_at":              self.created_at,
            "started_at":              self.started_at,
            "completed_at":            self.completed_at,
            "updated_at":              self.updated_at,
        }
