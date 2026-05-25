"""
app/models/comment.py
======================
MongoDB document model for a single YouTube comment — fully production-grade.

COLLECTION: comments

DESIGN NOTES:
  • text_hash enables O(1) edit detection on re-scrape without text comparison.
  • text_formatted preserves YouTube's rich-text runs (links, emoji, bold).
  • like_count_display keeps the raw API string ("1.2K") alongside the parsed int.
  • reply_to_comment_id is heuristic (from @mention parsing) and is ALWAYS
    flagged with reply_link_type so callers know it's a guess.
  • Every timestamp field has a companion *_precision field because YouTube
    only gives relative strings ("2 years ago") — precision degrades for old
    comments and callers must know that.
  • status tracks the comment lifecycle across re-scrapes.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class CommentType:
    STANDARD      = "standard"
    SUPER_THANKS  = "super_thanks"   # paid "Super Thanks" comment
    MEMBERS_ONLY  = "members_only"   # visible to channel members only


class ReplyLinkType:
    API        = "api"          # parent_comment_id from API (reliable)
    HEURISTIC  = "heuristic"    # reply_to_comment_id from @mention parsing
    UNKNOWN    = "unknown"      # could not determine threading


class CommentStatus:
    ACTIVE      = "active"       # visible in the latest scrape
    NOT_VISIBLE = "not_visible"  # not seen on last scrape — may be deleted


class TimestampPrecision:
    EXACT       = "exact"     # full datetime known
    DAY         = "day"       # accurate to the day
    WEEK        = "week"
    MONTH       = "month"
    YEAR        = "year"
    APPROXIMATE = "approximate"  # only "X years ago" available


@dataclass
class CommentDocument:
    # ── Core identity (immutable) ──────────────────────────────────────────
    comment_id: str          # YouTube comment ID  e.g. "UgzABC123"
    video_id:   str

    # ── Thread position (immutable after first scrape) ────────────────────
    is_reply:           bool          = False
    parent_comment_id:  Optional[str] = None   # TLC's comment_id (from API — reliable)
    reply_to_comment_id:Optional[str] = None   # which reply this responds to (heuristic)
    reply_link_type:    str           = ReplyLinkType.API
    thread_depth:       int           = 0      # 0 = TLC, 1 = reply (YouTube max via API)

    # ── Content (mutable — edit-tracked) ──────────────────────────────────
    text:           str  = ""
    text_formatted: list = field(default_factory=list)   # raw runs array from API
    text_hash:      str  = ""    # sha256(text) — set automatically via set_text()
    is_truncated:   bool = False # True when YT's "Read more" was not expanded

    # Edit tracking
    version:          int           = 1      # increments on each detected edit
    is_edited:        bool          = False
    edit_detected_at: Optional[datetime] = None   # when WE noticed the change

    # ── Author (semi-mutable — names/badges can change) ───────────────────
    author_name:             Optional[str]  = None
    author_channel_id:       Optional[str]  = None   # stable UCxxx identifier
    author_is_channel_owner: bool           = False
    author_is_verified:      bool           = False
    author_is_member:        bool           = False  # paid channel membership

    # ── Engagement (mutable — changes constantly) ─────────────────────────
    like_count:         int  = 0
    like_count_display: str  = "0"   # raw string from API ("1.2K")
    like_count_exact:   bool = True  # False when API returned abbreviated string
    reply_count:        int  = 0
    is_pinned:          bool = False
    is_hearted:         bool = False   # creator "hearted" this comment

    # ── Special comment type ───────────────────────────────────────────────
    comment_type:        str           = CommentType.STANDARD
    super_thanks_amount: Optional[str] = None   # "$5.00" if super_thanks

    # ── Timestamps ────────────────────────────────────────────────────────
    # Always store the raw string — it is the only guaranteed-available value.
    published_time_text:  Optional[str]      = None   # "2 years ago"
    published_at_approx:  Optional[datetime] = None   # computed estimate
    published_at_precision: str              = TimestampPrecision.APPROXIMATE
    scraped_at:           datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Scrape lifecycle
    first_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Scrape provenance ──────────────────────────────────────────────────
    scrape_job_id:      Optional[str] = None
    scrape_batch_number:Optional[int] = None
    scrape_sort_order:  str           = "newest_first"
    scrape_page_number: int           = 0

    # ── Lifecycle status ───────────────────────────────────────────────────
    status: str = CommentStatus.ACTIVE

    # ── MongoDB _id ───────────────────────────────────────────────────────
    _id: Optional[str] = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def set_text(self, text: str) -> None:
        """Set text and auto-compute its SHA-256 hash."""
        self.text      = text
        self.text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            # Identity
            "comment_id":              self.comment_id,
            "video_id":                self.video_id,
            # Thread
            "is_reply":                self.is_reply,
            "parent_comment_id":       self.parent_comment_id,
            "reply_to_comment_id":     self.reply_to_comment_id,
            "reply_link_type":         self.reply_link_type,
            "thread_depth":            self.thread_depth,
            # Content
            "text":                    self.text,
            "text_formatted":          self.text_formatted,
            "text_hash":               self.text_hash,
            "is_truncated":            self.is_truncated,
            "version":                 self.version,
            "is_edited":               self.is_edited,
            "edit_detected_at":        self.edit_detected_at,
            # Author
            "author_name":             self.author_name,
            "author_channel_id":       self.author_channel_id,
            "author_is_channel_owner": self.author_is_channel_owner,
            "author_is_verified":      self.author_is_verified,
            "author_is_member":        self.author_is_member,
            # Engagement
            "like_count":              self.like_count,
            "like_count_display":      self.like_count_display,
            "like_count_exact":        self.like_count_exact,
            "reply_count":             self.reply_count,
            "is_pinned":               self.is_pinned,
            "is_hearted":              self.is_hearted,
            # Type
            "comment_type":            self.comment_type,
            "super_thanks_amount":     self.super_thanks_amount,
            # Timestamps
            "published_time_text":     self.published_time_text,
            "published_at_approx":     self.published_at_approx,
            "published_at_precision":  self.published_at_precision,
            "scraped_at":              self.scraped_at,
            "first_seen_at":           self.first_seen_at,
            "last_seen_at":            self.last_seen_at,
            # Provenance
            "scrape_job_id":           self.scrape_job_id,
            "scrape_batch_number":     self.scrape_batch_number,
            "scrape_sort_order":       self.scrape_sort_order,
            "scrape_page_number":      self.scrape_page_number,
            # Status
            "status":                  self.status,
        }
