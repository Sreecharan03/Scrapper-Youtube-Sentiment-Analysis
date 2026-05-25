"""
app/models/scrape_session.py
=============================
Live state for an active scrape job — the "hot" working data.

COLLECTION: scrape_sessions
ONE document per job (upserted, not versioned).

WHY TWO STORES (Redis + MongoDB):
  Redis  — sub-millisecond reads during tight scraping loops
  MongoDB — durable copy survives a Redis flush or pod restart

The session document in MongoDB is the authoritative recovery source.
On worker startup, if Redis is empty, the worker reads from here and
rebuilds the Redis keys before resuming.

FIELDS EXPLAINED:
  current_tlc_token   The continuation token to use for the NEXT API call.
                      Saved after every successful 100-comment sub-batch write —
                      not after every API call (to avoid write amplification).
  token_obtained_at   Approximate time this token was issued by YouTube.
                      YouTube tokens expire ~6 hours after issue; if
                      (now - token_obtained_at) > 5h we proactively refresh
                      before starting a new batch rather than failing mid-batch.
  sub_batch_number    Running count of 100-comment writes completed this session.
  comments_written_total  Cumulative comments written across ALL batches for this job.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ScrapeSessionDocument:
    # ── Identity ─────────────────────────────────────────────────────────
    job_id:   str          # unique — one session per job
    video_id: str

    # ── Continuation state ───────────────────────────────────────────────
    current_tlc_token: str
    token_obtained_at: datetime

    # ── Progress counters ────────────────────────────────────────────────
    sub_batch_number:       int = 0
    comments_written_total: int = 0
    current_batch_number:   int = 1

    # ── Checkpoint timestamp ─────────────────────────────────────────────
    last_checkpoint_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── MongoDB _id ──────────────────────────────────────────────────────
    _id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id":                self.job_id,
            "video_id":              self.video_id,
            "current_tlc_token":     self.current_tlc_token,
            "token_obtained_at":     self.token_obtained_at,
            "sub_batch_number":      self.sub_batch_number,
            "comments_written_total":self.comments_written_total,
            "current_batch_number":  self.current_batch_number,
            "last_checkpoint_at":    self.last_checkpoint_at,
        }
