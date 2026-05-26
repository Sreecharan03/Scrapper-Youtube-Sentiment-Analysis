"""
app/models/transcript.py
========================
MongoDB document model for a video transcript.

STORAGE DESIGN:
  One document per video (+ language combo if multiple fetched).
  Segments are stored inline as an array — a 2-hour video produces ~1,500
  segments (~150 KB), well within MongoDB's 16 MB document limit.

MULTI-LANGUAGE STRATEGY:
  - original_segments: raw text in whatever language the video has captions
  - english_segments:  YouTube-translated version (set only when original != en)
  - Downstream NLP (sentiment, clustering) always works on english_segments
    when available, falls back to original_segments.
  - available_languages: full list returned by youtube-transcript-api so the
    caller can re-fetch in a different language without a round-trip to YouTube.

STATUS LIFECYCLE:
  pending → fetching → completed
                    → unavailable   (video has no captions at all)
                    → failed        (network/parse error)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ── Status constants ───────────────────────────────────────────────────────

class TranscriptStatus:
    PENDING     = "pending"
    FETCHING    = "fetching"
    COMPLETED   = "completed"
    UNAVAILABLE = "unavailable"   # TranscriptsDisabled or no tracks at all
    FAILED      = "failed"


# ── Sub-document types (stored inline) ────────────────────────────────────

@dataclass
class TranscriptSegment:
    """One caption line with millisecond timestamps."""
    start_ms: int    # start of this line in milliseconds from video start
    end_ms:   int    # end of this line in milliseconds
    text:     str    # raw caption text (may contain HTML entities from YT)

    def to_dict(self) -> dict:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "text": self.text}


@dataclass
class AvailableLanguage:
    """One entry in the list of caption tracks YouTube provides."""
    language_code:  str   # ISO 639-1 code, e.g. "en", "hi", "es"
    language_name:  str   # Human-readable name, e.g. "English (auto-generated)"
    is_generated:   bool  # True = YouTube auto-captions, False = manually uploaded

    def to_dict(self) -> dict:
        return {
            "language_code": self.language_code,
            "language_name": self.language_name,
            "is_generated":  self.is_generated,
        }


# ── Main document ─────────────────────────────────────────────────────────

@dataclass
class TranscriptDocument:
    """
    Represents one transcript fetch result stored in the `transcripts` collection.
    Uniquely identified by video_id (one transcript per video — re-fetch overwrites).
    """

    # ── Identity ──────────────────────────────────────────────────────────
    video_id: str

    # ── Status ────────────────────────────────────────────────────────────
    status:     str  = TranscriptStatus.PENDING
    error:      Optional[str] = None   # populated when status=failed

    # ── Language metadata ─────────────────────────────────────────────────
    original_language_code: Optional[str] = None   # e.g. "hi"
    original_language_name: Optional[str] = None   # e.g. "Hindi (auto-generated)"
    is_auto_generated:       Optional[bool] = None  # True = YouTube auto-captions
    available_languages:     list = field(default_factory=list)
    # [{language_code, language_name, is_generated}, ...]

    # ── Segments ──────────────────────────────────────────────────────────
    original_segments:  list = field(default_factory=list)
    # [{start_ms, end_ms, text}, ...] — in original_language

    english_segments:   Optional[list] = None
    # Same structure but YouTube-translated to English.
    # None when original_language_code == "en" (already English, no translation needed).
    # None when translation not available or failed.

    is_translated:      bool = False
    # True if english_segments was populated via YouTube's translation API.

    # ── Stats ─────────────────────────────────────────────────────────────
    segment_count:        int   = 0
    total_duration_secs:  float = 0.0

    # ── Timestamps ────────────────────────────────────────────────────────
    fetched_at:   Optional[datetime] = None
    created_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── MongoDB _id (set after insert) ────────────────────────────────────
    _id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "video_id":               self.video_id,
            "status":                 self.status,
            "error":                  self.error,
            "original_language_code": self.original_language_code,
            "original_language_name": self.original_language_name,
            "is_auto_generated":      self.is_auto_generated,
            "available_languages":    self.available_languages,
            "original_segments":      self.original_segments,
            "english_segments":       self.english_segments,
            "is_translated":          self.is_translated,
            "segment_count":          self.segment_count,
            "total_duration_secs":    self.total_duration_secs,
            "fetched_at":             self.fetched_at,
            "created_at":             self.created_at,
            "updated_at":             self.updated_at,
        }
