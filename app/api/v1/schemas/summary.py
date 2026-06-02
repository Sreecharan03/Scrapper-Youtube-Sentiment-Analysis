"""
app/api/v1/schemas/summary.py
================================
Pydantic response schemas for the Summary API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SummaryResponse(BaseModel):
    video_id:    str
    status:      str
    error:       Optional[str] = None

    # Core LLM output
    overview:             Optional[str]       = None
    key_topics:           List[Dict[str, Any]] = Field(default_factory=list)
    key_claims:           List[str]            = Field(default_factory=list)
    emotional_arc:        List[Dict[str, Any]] = Field(default_factory=list)
    named_entities:       Optional[Dict]       = None
    controversy_triggers: List[str]            = Field(default_factory=list)
    video_promises:       List[str]            = Field(default_factory=list)
    humor_moments:        List[Dict[str, Any]] = Field(default_factory=list)
    audience_signals:     Optional[Dict]       = None
    content_warnings:     List[str]            = Field(default_factory=list)
    tone:                 Optional[str]        = None
    content_type:         Optional[str]        = None

    # Generation metadata
    model:              Optional[str] = None
    critique_severity:  Optional[str] = None
    total_input_tokens: int           = 0
    total_output_tokens:int           = 0

    @classmethod
    def from_document(cls, doc: dict) -> "SummaryResponse":
        return cls(
            video_id              = doc["video_id"],
            status                = doc.get("status", "pending"),
            error                 = doc.get("error"),
            overview              = doc.get("overview"),
            key_topics            = doc.get("key_topics") or [],
            key_claims            = doc.get("key_claims") or [],
            emotional_arc         = doc.get("emotional_arc") or [],
            named_entities        = doc.get("named_entities"),
            controversy_triggers  = doc.get("controversy_triggers") or [],
            video_promises        = doc.get("video_promises") or [],
            humor_moments         = doc.get("humor_moments") or [],
            audience_signals      = doc.get("audience_signals"),
            content_warnings      = doc.get("content_warnings") or [],
            tone                  = doc.get("tone"),
            content_type          = doc.get("content_type"),
            model                 = doc.get("model"),
            critique_severity     = doc.get("critique_severity"),
            total_input_tokens    = doc.get("total_input_tokens", 0),
            total_output_tokens   = doc.get("total_output_tokens", 0),
        )


class GenerateSummaryResponse(BaseModel):
    video_id: str
    status:   str = "pending"
    message:  str
