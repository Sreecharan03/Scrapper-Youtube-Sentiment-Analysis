"""
app/models/failed_reply.py
===========================
Tracks reply continuation tokens that exhausted all retry attempts.

COLLECTION: failed_replies
WHY: Reply failures must NOT block the TLC phase.  Instead, tokens that
fail permanently are written here so a separate recovery job can retry
them later without touching the main job flow.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class FailedReplyStatus:
    PENDING_RETRY = "pending_retry"   # waiting to be retried
    EXHAUSTED     = "exhausted"       # gave up, needs manual review


@dataclass
class FailedReplyDocument:
    # ── Identity ─────────────────────────────────────────────────────────
    job_id:     str
    video_id:   str
    comment_id: str    # the TLC whose replies failed to load
    reply_token: str   # the continuation token that kept failing

    # ── Failure detail ───────────────────────────────────────────────────
    attempts:          int = 0
    last_error:        Optional[str] = None
    last_error_type:   Optional[str] = None   # exception class name
    last_attempt_at:   Optional[datetime] = None

    # ── Status ───────────────────────────────────────────────────────────
    status: str = FailedReplyStatus.PENDING_RETRY

    # ── Timestamps ───────────────────────────────────────────────────────
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── MongoDB _id ──────────────────────────────────────────────────────
    _id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id":           self.job_id,
            "video_id":         self.video_id,
            "comment_id":       self.comment_id,
            "reply_token":      self.reply_token,
            "attempts":         self.attempts,
            "last_error":       self.last_error,
            "last_error_type":  self.last_error_type,
            "last_attempt_at":  self.last_attempt_at,
            "status":           self.status,
            "created_at":       self.created_at,
        }
