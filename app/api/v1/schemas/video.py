"""
app/api/v1/schemas/video.py
============================
Pydantic models for Video API responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class VideoResponse(BaseModel):
    """Video metadata as returned by the API."""

    video_id: str
    url: str
    title: Optional[str] = None
    channel_name: Optional[str] = None
    view_count: Optional[int] = None
    comment_count: Optional[int] = None
    comments_scraped: int = 0
    scrape_completed: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, doc: dict) -> "VideoResponse":
        return cls(
            video_id=doc["video_id"],
            url=doc["url"],
            title=doc.get("title"),
            channel_name=doc.get("channel_name"),
            view_count=doc.get("view_count"),
            comment_count=doc.get("comment_count"),
            comments_scraped=doc.get("comments_scraped", 0),
            scrape_completed=doc.get("scrape_completed", False),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )
