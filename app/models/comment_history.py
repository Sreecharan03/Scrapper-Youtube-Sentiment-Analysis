"""
app/models/comment_history.py
===============================
Append-only log of every detected edit to a comment.

COLLECTION: comment_history
WRITE PATTERN: insert-only — never updated, never deleted.

WHY SEPARATE FROM comments:
  The main `comments` collection always holds the *current* state.
  This collection holds every *previous* state archived at the moment
  an edit was detected.  Keeping them separate means:
    - Main comment queries stay fast (no version bloat)
    - Full edit history is available via a secondary query
    - Storage can be tiered/archived independently
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class CommentHistoryDocument:
    # ── Identity ─────────────────────────────────────────────────────────
    comment_id: str
    video_id:   str

    # ── Archived version ─────────────────────────────────────────────────
    # `version` is the version number being ARCHIVED (the old one).
    # After this insert the comment document's version becomes version + 1.
    version:   int
    text:      str        # the old text
    text_hash: str        # sha256 of old text (for verification)

    # ── Engagement snapshot at detection time ────────────────────────────
    like_count_at_detection: int = 0

    # ── Provenance ───────────────────────────────────────────────────────
    detected_at:        datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detected_by_job_id: Optional[str] = None   # which re-scrape job noticed the change

    # ── MongoDB _id ──────────────────────────────────────────────────────
    _id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "comment_id":              self.comment_id,
            "video_id":                self.video_id,
            "version":                 self.version,
            "text":                    self.text,
            "text_hash":               self.text_hash,
            "like_count_at_detection": self.like_count_at_detection,
            "detected_at":             self.detected_at,
            "detected_by_job_id":      self.detected_by_job_id,
        }
