from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class IntentSummaryItem(BaseModel):
    summary: str
    count:   int


class IntentSummaryTriggerResponse(BaseModel):
    video_id: str
    status:   str
    message:  str


class IntentSummariesResponse(BaseModel):
    video_id:         str
    status:           str
    generated_at:     Optional[datetime]
    overall_summary:  str
    intent_summaries: dict[str, IntentSummaryItem]
    error:            Optional[str] = None
