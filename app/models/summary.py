"""
app/models/summary.py
======================
MongoDB document model for a video LLM summary.

Stored separately from transcripts so summaries can be regenerated
independently (e.g., with a new prompt) without touching the transcript doc.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class SummaryStatus:
    PENDING    = "pending"
    GENERATING = "generating"
    COMPLETED  = "completed"
    FAILED     = "failed"


@dataclass
class SummaryDocument:
    video_id: str

    status:  str            = SummaryStatus.PENDING
    error:   Optional[str]  = None

    # ── LLM output fields (populated on completion) ───────────────────────
    overview:             Optional[str]  = None
    key_topics:           list           = field(default_factory=list)
    key_claims:           list           = field(default_factory=list)
    emotional_arc:        list           = field(default_factory=list)
    named_entities:       Optional[dict] = None
    controversy_triggers: list           = field(default_factory=list)
    video_promises:       list           = field(default_factory=list)
    humor_moments:        list           = field(default_factory=list)
    audience_signals:     Optional[dict] = None
    content_warnings:     list           = field(default_factory=list)
    tone:                 Optional[str]  = None
    content_type:         Optional[str]  = None

    # ── Generation metadata ───────────────────────────────────────────────
    model:                Optional[str]  = None
    critique_severity:    Optional[str]  = None   # low | medium | high
    critique_notes:       Optional[str]  = None
    total_input_tokens:   int            = 0
    total_output_tokens:  int            = 0

    # ── Timestamps ────────────────────────────────────────────────────────
    generated_at: Optional[datetime] = None
    created_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "video_id":            self.video_id,
            "status":              self.status,
            "error":               self.error,
            "overview":            self.overview,
            "key_topics":          self.key_topics,
            "key_claims":          self.key_claims,
            "emotional_arc":       self.emotional_arc,
            "named_entities":      self.named_entities,
            "controversy_triggers":self.controversy_triggers,
            "video_promises":      self.video_promises,
            "humor_moments":       self.humor_moments,
            "audience_signals":    self.audience_signals,
            "content_warnings":    self.content_warnings,
            "tone":                self.tone,
            "content_type":        self.content_type,
            "model":               self.model,
            "critique_severity":   self.critique_severity,
            "critique_notes":      self.critique_notes,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "generated_at":        self.generated_at,
            "created_at":          self.created_at,
            "updated_at":          self.updated_at,
        }
