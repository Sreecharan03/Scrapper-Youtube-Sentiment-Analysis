"""
app/api/v1/schemas/analysis.py
================================
Request/response schemas for the comment classification and analysis endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ClassifyCommentsResponse(BaseModel):
    video_id: str
    status:   str
    message:  str


class SentimentBucket(BaseModel):
    count: int
    pct:   float


class IntentBucket(BaseModel):
    count: int
    pct:   float


class SentimentBreakdown(BaseModel):
    positive: SentimentBucket
    neutral:  SentimentBucket
    negative: SentimentBucket


class IntentBreakdown(BaseModel):
    question:     IntentBucket
    praise:       IntentBucket
    criticism:    IntentBucket
    confusion:    IntentBucket
    misconception: IntentBucket
    request:      IntentBucket
    spam:         IntentBucket
    off_topic:    IntentBucket


class CommentAnalysisResponse(BaseModel):
    video_id:                str
    status:                  str
    total_comments:          Optional[int]         = None
    classified_count:        Optional[int]         = None
    failed_count:            Optional[int]         = None
    skipped_count:           Optional[int]         = None
    sentiment_breakdown:     Optional[dict]        = None
    intent_breakdown:        Optional[dict]        = None
    computed_at:             Optional[datetime]    = None
    classification_version:  Optional[str]         = None
    error:                   Optional[str]         = None

    @classmethod
    def from_document(cls, doc: dict) -> "CommentAnalysisResponse":
        return cls(
            video_id               = doc["video_id"],
            status                 = doc.get("status", "unknown"),
            total_comments         = doc.get("total_comments"),
            classified_count       = doc.get("classified_count"),
            failed_count           = doc.get("failed_count"),
            skipped_count          = doc.get("skipped_count"),
            sentiment_breakdown    = doc.get("sentiment_breakdown"),
            intent_breakdown       = doc.get("intent_breakdown"),
            computed_at            = doc.get("computed_at"),
            classification_version = doc.get("classification_version"),
            error                  = doc.get("error"),
        )
