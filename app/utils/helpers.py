"""
app/utils/helpers.py
=====================
Shared stateless utility functions.

RULES FOR THIS FILE:
  - Pure functions only — no I/O, no imports from app.db or app.workers.
  - If a helper needs a DB connection, it belongs in a repository, not here.
  - These functions must be trivially unit-testable with no mocking.
"""

import re
from datetime import datetime, timezone
from typing import Optional


# ------------------------------------------------------------------ #
# YouTube URL / ID utilities                                           #
# ------------------------------------------------------------------ #

_VIDEO_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the 11-character YouTube video ID from any YouTube URL format.

    Supports:
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - https://youtu.be/dQw4w9WgXcQ
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s

    Returns:
        11-char video ID string, or None if not found.
    """
    if not url:
        return None
    match = _VIDEO_ID_PATTERN.search(url)
    return match.group(1) if match else None


def build_video_url(video_id: str) -> str:
    """Return the canonical YouTube URL for a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


def is_valid_video_id(video_id: str) -> bool:
    """Validate that a string looks like a YouTube video ID (11 alphanumeric chars)."""
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""))


# ------------------------------------------------------------------ #
# DateTime utilities                                                   #
# ------------------------------------------------------------------ #

def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def parse_yt_timestamp(raw: Optional[str]) -> Optional[datetime]:
    """
    Parse YouTube's relative timestamp strings into datetime objects.

    YouTube internal APIs return strings like "3 years ago", "2 months ago".
    This is a stub — Phase 2 will implement the full parser when we have
    real API response examples to work from.

    Returns:
        Parsed datetime, or None if parsing fails.
    """
    if not raw:
        return None
    # Phase 2: implement full parser
    # For now, return None rather than crash
    return None


# ------------------------------------------------------------------ #
# Text utilities                                                       #
# ------------------------------------------------------------------ #

def clean_comment_text(text: str) -> str:
    """
    Normalise raw comment text:
      - Strip leading/trailing whitespace
      - Collapse multiple consecutive blank lines into one
      - Preserve emojis and Unicode (do NOT strip non-ASCII)
    """
    if not text:
        return ""
    text = text.strip()
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text for log display. Not for storing — always store full text."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
