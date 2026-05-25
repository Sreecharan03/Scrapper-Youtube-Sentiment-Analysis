"""
app/models/video.py
===================
MongoDB document model for a YouTube video record.

WHY SEPARATE FROM api/v1/schemas/:
  - This defines what gets STORED in MongoDB.
  - The API schema defines what gets RETURNED to callers.
  - They evolve independently: you might store raw scraped fields but
    return only a clean subset to API consumers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class VideoDocument:
    """
    Represents a YouTube video stored in the `videos` collection.

    Fields prefixed with `scraped_` hold raw data from the YT internal API.
    Fields without prefix are derived/normalised.
    """

    # ---- Identity ----
    video_id: str                    # YouTube video ID (e.g. "dQw4w9WgXcQ")
    url: str                         # Full canonical URL

    # ---- Metadata (populated after scrape) ----
    title: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    published_at: Optional[datetime] = None
    description: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None

    # ---- Scrape tracking ----
    # How many comments have we actually collected so far?
    comments_scraped: int = 0
    # Has scraping reached the end of the comment list?
    scrape_completed: bool = False
    # Continuation token for resuming an interrupted scrape
    last_continuation_token: Optional[str] = None

    # ---- Timestamps ----
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ---- MongoDB document ID (set after insert) ----
    # Using str instead of ObjectId keeps the model free of pymongo imports.
    # Repositories handle the ObjectId conversion.
    _id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise for MongoDB insertion. Excludes None _id (let Atlas generate it)."""
        data = {
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "published_at": self.published_at,
            "description": self.description,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "comments_scraped": self.comments_scraped,
            "scrape_completed": self.scrape_completed,
            "last_continuation_token": self.last_continuation_token,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return data
