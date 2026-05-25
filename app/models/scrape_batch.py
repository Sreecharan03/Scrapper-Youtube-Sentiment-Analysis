"""
app/models/scrape_batch.py
===========================
MongoDB document model for a single 5 000-comment batch within a scrape job.

COLLECTION: scrape_batches
One document per batch per job.  A job with 100 k comments produces ~20 documents
here.  Keeping batch records separate from the job document prevents the job
document from bloating with embedded arrays.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class BatchStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    PAUSED    = "paused"      # DB write failed after all retries — needs operator action


@dataclass
class ScrapeBatchDocument:
    # ── Identity ────────────────────────────────────────────────────────
    job_id:        str
    batch_number:  int          # 1-based, sequential within the job

    # ── Token bookmarks ──────────────────────────────────────────────────
    # token_at_start is saved *before* any API calls so a crashed batch
    # can restart from the exact same point.
    token_at_start: str
    token_at_end:   Optional[str] = None   # set on COMPLETED

    # ── Progress ─────────────────────────────────────────────────────────
    comments_written:   int = 0
    duplicates_skipped: int = 0   # unique-index violations silently caught
    reply_tokens_found: int = 0   # TLCs with replies discovered in this batch
    sub_batches_done:   int = 0   # number of 100-comment sub-batches completed
    sub_batches_total:  int = 50  # target (5 000 / 100)

    # ── Status ───────────────────────────────────────────────────────────
    status: str = BatchStatus.PENDING

    # ── Runtime metadata ─────────────────────────────────────────────────
    celery_task_id:   Optional[str] = None
    worker_hostname:  Optional[str] = None
    errors:           list = field(default_factory=list)   # transient errors that were recovered

    # ── Timestamps ───────────────────────────────────────────────────────
    created_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at:   Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # ── MongoDB _id ──────────────────────────────────────────────────────
    _id: Optional[str] = None

    # ── Helpers ──────────────────────────────────────────────────────────
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def progress_pct(self) -> float:
        target = self.sub_batches_total or 50
        return round((self.sub_batches_done / target) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "job_id":            self.job_id,
            "batch_number":      self.batch_number,
            "token_at_start":    self.token_at_start,
            "token_at_end":      self.token_at_end,
            "comments_written":  self.comments_written,
            "duplicates_skipped":self.duplicates_skipped,
            "reply_tokens_found":self.reply_tokens_found,
            "sub_batches_done":  self.sub_batches_done,
            "sub_batches_total": self.sub_batches_total,
            "status":            self.status,
            "celery_task_id":    self.celery_task_id,
            "worker_hostname":   self.worker_hostname,
            "errors":            self.errors,
            "created_at":        self.created_at,
            "started_at":        self.started_at,
            "completed_at":      self.completed_at,
        }
