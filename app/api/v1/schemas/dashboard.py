"""
app/api/v1/schemas/dashboard.py
================================
Response schema for GET /api/v1/dashboard/{video_id}.

Designed around the Sighnal UI hierarchy:
  1. Video header
  2. Scores (health / opportunity / risk)
  3. Executive summary + stage/mood chips
  4. Top 3 video ideas
  5. Intent breakdown + per-intent summaries (for tabs)
  6. Deep dive (gaps, misconceptions, controversies, unanswered, series, alerts)
  7. Pipeline status (tells frontend which panels are ready)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ── Sub-models ────────────────────────────────────────────────────────────────

class VideoMeta(BaseModel):
    video_id:           str
    url:                str
    title:              Optional[str]         = None
    channel_name:       Optional[str]         = None
    view_count:         Optional[int]         = None
    like_count:         Optional[int]         = None
    comments_collected: int                   = 0


class Scores(BaseModel):
    health:      int  # 0-100: audience positivity weighted against risk signals
    opportunity: int  # 0-100: demand strength from top video idea's demand_score
    risk:        int  # 0-100: criticism + misconception pressure


class IntentBucket(BaseModel):
    count: int
    pct:   float


class IntentBreakdown(BaseModel):
    question:      IntentBucket = IntentBucket(count=0, pct=0)
    praise:        IntentBucket = IntentBucket(count=0, pct=0)
    criticism:     IntentBucket = IntentBucket(count=0, pct=0)
    confusion:     IntentBucket = IntentBucket(count=0, pct=0)
    misconception: IntentBucket = IntentBucket(count=0, pct=0)
    request:       IntentBucket = IntentBucket(count=0, pct=0)
    spam:          IntentBucket = IntentBucket(count=0, pct=0)
    off_topic:     IntentBucket = IntentBucket(count=0, pct=0)
    total:         int          = 0


class SentimentBucket(BaseModel):
    count: int
    pct:   float


class SentimentBreakdown(BaseModel):
    positive: SentimentBucket = SentimentBucket(count=0, pct=0)
    neutral:  SentimentBucket = SentimentBucket(count=0, pct=0)
    negative: SentimentBucket = SentimentBucket(count=0, pct=0)


class IntentTabItem(BaseModel):
    intent:  str
    count:   int
    summary: str  # from intent_summaries — the AI narrative for that intent


class VideoIdeaCard(BaseModel):
    rank:           int
    title:          str
    demand_score:   int
    confidence_pct: int
    why:            str
    evidence_count: int
    format:         str


class TopComment(BaseModel):
    comment_id: str
    text:       str
    like_count: int


class ContentGapCard(BaseModel):
    cluster_id:     int
    label:          str
    comment_count:  int
    question_pct:   float
    priority_score: float
    what_to_do:     str
    suggested_hook: str
    urgency:        str
    top_comments:   list[TopComment] = []


class MisconceptionCard(BaseModel):
    cluster_id:          int
    cluster_label:       str
    misconception_count: int
    related_claim:       str
    what_to_do:          str
    suggested_hook:      str
    urgency:             str
    top_comments:        list[TopComment] = []


class ControversyCard(BaseModel):
    cluster_id:      int
    cluster_label:   str
    criticism_count: int
    criticism_pct:   float
    matched_trigger: str
    what_to_do:      str
    suggested_hook:  str
    urgency:         str
    top_comments:    list[TopComment] = []


class UnansweredCard(BaseModel):
    comment_id:    str
    text:          str
    like_count:    int
    cluster_label: str


class PipelineStatus(BaseModel):
    classification:  str  # completed | processing | failed | missing
    clustering:      str
    recommendations: str
    intent_summaries: str


# ── Top-level response ────────────────────────────────────────────────────────

class DashboardResponse(BaseModel):
    video:              VideoMeta
    scores:             Scores
    # Section 2 — AI executive brief
    executive_summary:  str               = ""
    audience_stage:     str               = ""
    audience_mood:      str               = ""
    overall_summary:    str               = ""   # from intent_summaries
    # Section 3 — Top video ideas (top 3 for hero panel)
    top_video_ideas:    list[VideoIdeaCard] = []
    # Section 4 — Intent breakdown + tab summaries
    intent_breakdown:   IntentBreakdown   = IntentBreakdown()
    sentiment_breakdown: SentimentBreakdown = SentimentBreakdown()
    intent_tabs:        list[IntentTabItem] = []
    # Section 5 — Deep dive
    content_gaps:           list[ContentGapCard]     = []
    misconceptions:         list[MisconceptionCard]  = []
    controversy_hotspots:   list[ControversyCard]    = []
    unanswered_questions:   list[UnansweredCard]     = []
    purchase_intent_signals: list[str]               = []
    content_series:         list[str]                = []
    risk_alerts:            list[str]                = []
    # Meta
    pipeline_status:    PipelineStatus
    generated_at:       Optional[datetime] = None
