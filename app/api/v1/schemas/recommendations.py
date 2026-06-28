"""
app/api/v1/schemas/recommendations.py
=======================================
Pydantic v2 schemas for Phase 3D recommendation endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TopComment(BaseModel):
    comment_id: str
    text:       str
    like_count: int


class ContentGapItem(BaseModel):
    cluster_id:     int
    label:          str
    priority_score: float
    comment_count:  int
    question_pct:   float
    gap_sim:        float
    keywords:       list[str]
    top_comments:   list[TopComment]
    what_to_do:     str
    why:            str
    suggested_hook: str
    urgency:        str
    impact_type:    str


class MisconceptionItem(BaseModel):
    cluster_id:          int
    cluster_label:       str
    misconception_count: int
    related_claim:       str
    top_comments:        list[TopComment]
    what_to_do:          str
    why:                 str
    suggested_hook:      str
    urgency:             str
    impact_type:         str


class ControversyItem(BaseModel):
    cluster_id:      int
    cluster_label:   str
    criticism_count: int
    criticism_pct:   float
    matched_trigger: str
    sentiment:       str
    top_comments:    list[TopComment]
    what_to_do:      str
    why:             str
    suggested_hook:  str
    urgency:         str
    impact_type:     str


class UnansweredItem(BaseModel):
    comment_id:    str
    text:          str
    like_count:    int
    cluster_id:    int
    cluster_label: str


class VideoIdeaItem(BaseModel):
    rank:           int
    title:          str
    demand_score:   int
    confidence_pct: int
    why:            str
    evidence_count: int
    format:         str


class RecommendationTriggerResponse(BaseModel):
    video_id: str
    status:   str
    message:  str


class RecommendationsResponse(BaseModel):
    video_id:               str
    status:                 str
    generated_at:           Optional[datetime]
    # Strategic layer
    executive_summary:       str                  = ""
    audience_stage:          str                  = ""
    audience_mood:           str                  = ""
    top_video_ideas:         list[VideoIdeaItem]  = []
    purchase_intent_signals: list[str]            = []
    content_series:          list[str]            = []
    risk_alerts:             list[str]            = []
    # Analysis types
    content_gaps:            list[ContentGapItem]
    misconceptions:          list[MisconceptionItem]
    controversy_hotspots:    list[ControversyItem]
    unanswered_questions:    list[UnansweredItem]
    error:                   Optional[str]        = None
