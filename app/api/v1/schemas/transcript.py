"""
app/api/v1/schemas/transcript.py
==================================
Pydantic schemas for the Transcript API.

SEPARATION FROM models/transcript.py:
  models/transcript.py  → what is STORED in MongoDB (dataclass, raw dict)
  schemas/transcript.py → what is RETURNED to API callers (Pydantic, validated)

  The API response omits internal fields (created_at, updated_at) and
  keeps segment arrays optional — a "pending" response won't include them.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Sub-schemas ───────────────────────────────────────────────────────────

class SegmentSchema(BaseModel):
    start_ms: int   = Field(..., description="Segment start in milliseconds from video start")
    end_ms:   int   = Field(..., description="Segment end in milliseconds")
    text:     str   = Field(..., description="Caption text for this segment")


class AvailableLanguageSchema(BaseModel):
    language_code: str  = Field(..., description="ISO 639-1 code, e.g. 'en', 'hi'")
    language_name: str  = Field(..., description="Human-readable name, e.g. 'Hindi (auto-generated)'")
    is_generated:  bool = Field(..., description="True if YouTube auto-generated captions")


# ── Response schemas ──────────────────────────────────────────────────────

class TranscriptResponse(BaseModel):
    """
    Returned by GET /transcripts/{video_id}.
    segments arrays are omitted when status != completed (no data yet).
    """
    video_id: str
    status:   str = Field(..., description="pending | fetching | completed | unavailable | failed")
    error:    Optional[str] = Field(None, description="Set when status is unavailable or failed")

    # Language info (populated once fetch completes)
    original_language_code: Optional[str] = Field(None, description="e.g. 'hi'")
    original_language_name: Optional[str] = Field(None, description="e.g. 'Hindi (auto-generated)'")
    is_auto_generated:       Optional[bool] = None
    is_translated:           bool = Field(False, description="True if english_segments is a YT translation")

    available_languages: List[AvailableLanguageSchema] = Field(
        default_factory=list,
        description="All caption tracks YouTube has for this video",
    )

    # Segments (populated once fetch completes; omitted in list/status calls)
    original_segments: Optional[List[SegmentSchema]] = Field(
        None,
        description="Transcript in original language",
    )
    english_segments: Optional[List[SegmentSchema]] = Field(
        None,
        description="YouTube-translated English segments (None if original is already English)",
    )

    # Stats
    segment_count:       int   = 0
    total_duration_secs: float = 0.0

    @classmethod
    def from_document(cls, doc: dict, include_segments: bool = True) -> "TranscriptResponse":
        """
        Build response from a raw MongoDB document.

        Args:
            doc:              Raw dict from TranscriptRepository.
            include_segments: Pass False to omit segment arrays (for status-only calls).
        """
        available = [
            AvailableLanguageSchema(
                language_code = lang["language_code"],
                language_name = lang["language_name"],
                is_generated  = lang["is_generated"],
            )
            for lang in (doc.get("available_languages") or [])
        ]

        def _parse_segments(raw: Optional[list]) -> Optional[List[SegmentSchema]]:
            if raw is None:
                return None
            return [SegmentSchema(start_ms=s["start_ms"], end_ms=s["end_ms"], text=s["text"]) for s in raw]

        return cls(
            video_id                = doc["video_id"],
            status                  = doc.get("status", "pending"),
            error                   = doc.get("error"),
            original_language_code  = doc.get("original_language_code"),
            original_language_name  = doc.get("original_language_name"),
            is_auto_generated       = doc.get("is_auto_generated"),
            is_translated           = doc.get("is_translated", False),
            available_languages     = available,
            original_segments       = _parse_segments(doc.get("original_segments")) if include_segments else None,
            english_segments        = _parse_segments(doc.get("english_segments"))  if include_segments else None,
            segment_count           = doc.get("segment_count", 0),
            total_duration_secs     = doc.get("total_duration_secs", 0.0),
        )


class TranscriptStatusResponse(BaseModel):
    """Lightweight status-only response (no segments). Used by GET status endpoint."""
    video_id:              str
    status:                str
    error:                 Optional[str]   = None
    original_language_code:Optional[str]   = None
    original_language_name:Optional[str]   = None
    is_auto_generated:     Optional[bool]  = None
    is_translated:         bool            = False
    available_languages:   List[AvailableLanguageSchema] = Field(default_factory=list)
    segment_count:         int             = 0
    total_duration_secs:   float           = 0.0

    @classmethod
    def from_document(cls, doc: dict) -> "TranscriptStatusResponse":
        return cls(**TranscriptResponse.from_document(doc, include_segments=False).model_dump(
            exclude={"original_segments", "english_segments"}
        ))


class FetchTranscriptRequest(BaseModel):
    """
    Request body for POST /transcripts/{video_id}.
    preferred_languages is a priority list — the fetcher tries each language
    in order before falling back to any available transcript.
    """
    preferred_languages: List[str] = Field(
        default=["en"],
        description=(
            "Priority-ordered list of ISO 639-1 language codes. "
            "Example: ['hi', 'en'] fetches Hindi if available, otherwise English. "
            "If none of the preferred languages exist, falls back to any available transcript."
        ),
        min_length=1,
        max_length=10,
    )


class FetchTranscriptResponse(BaseModel):
    """Returned immediately after POST /transcripts/{video_id} is accepted."""
    video_id:            str
    status:              str = "pending"
    preferred_languages: List[str]
    message:             str
