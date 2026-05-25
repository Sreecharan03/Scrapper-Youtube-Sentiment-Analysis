"""
app/api/v1/schemas/comment.py
==============================
Pydantic models for Comment API responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CommentResponse(BaseModel):
    """Single comment as returned by the API."""

    comment_id: str
    video_id: str
    text: str
    author_name: Optional[str] = None
    like_count: int = 0
    reply_count: int = 0
    is_pinned: bool = False
    is_hearted: bool = False
    is_reply: bool = False
    parent_comment_id: Optional[str] = None
    published_at: Optional[datetime] = None
    scraped_at: datetime

    @classmethod
    def from_document(cls, doc: dict) -> "CommentResponse":
        return cls(
            comment_id=doc["comment_id"],
            video_id=doc["video_id"],
            text=doc["text"],
            author_name=doc.get("author_name"),
            like_count=doc.get("like_count", 0),
            reply_count=doc.get("reply_count", 0),
            is_pinned=doc.get("is_pinned", False),
            is_hearted=doc.get("is_hearted", False),
            is_reply=doc.get("is_reply", False),
            parent_comment_id=doc.get("parent_comment_id"),
            published_at=doc.get("published_at"),
            scraped_at=doc["scraped_at"],
        )


class CommentListResponse(BaseModel):
    """Paginated list of comments."""

    comments: list[CommentResponse]
    video_id: str
    total_stored: int
    skip: int
    limit: int
