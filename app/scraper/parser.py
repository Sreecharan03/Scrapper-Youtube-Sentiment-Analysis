"""
app/scraper/parser.py
======================
YouTube API response parser — raw JSON → NormalizedComment.

DESIGN PRINCIPLES:
  1. NEVER crash on a missing field.  Use deep_get() everywhere.
  2. Log unexpected structures at DEBUG (raw fragment) for future debugging.
  3. Store raw strings when precision is ambiguous (timestamps, like counts).
  4. Heuristic reply-to reconstruction is clearly flagged.

YOUTUBE RESPONSE SHAPE (simplified):
  {
    "onResponseReceivedEndpoints": [
      {
        // Either key may appear:
        "appendContinuationItemsAction":  { "continuationItems": [...] }
        "reloadContinuationItemsCommand": { "continuationItems": [...] }
      }
    ]
  }

  Each item in continuationItems is either:
    { "commentThreadRenderer": { "comment": {...}, "replies": {...} } }
    { "commentRenderer":       {...} }               ← reply item
    { "continuationItemRenderer": {...} }            ← next-page token
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.logging import get_logger
from app.models.comment import (
    CommentDocument, CommentStatus, CommentType,
    ReplyLinkType, TimestampPrecision,
)
from app.scraper.constants import (
    YT_API_KEY_FALLBACK, YT_CLIENT_VERSION_FALLBACK,
)

logger = get_logger(__name__)


# ── Internal scraper-only dataclass ───────────────────────────────────────

@dataclass
class ParsedPage:
    """Result of parsing one YouTube API response page."""
    comments:        list[CommentDocument] = field(default_factory=list)
    next_token:      Optional[str]         = None   # continuation for next TLC page
    reply_tokens:    list[dict]            = field(default_factory=list)
    # [{comment_id, video_id, token}]  — TLCs with replies
    is_last_page:    bool                  = False   # no next_token found


# ── Safe nested-dict accessor ──────────────────────────────────────────────

def deep_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely traverse a nested dict/list.
    deep_get(d, "a", "b", "c") == d.get("a", {}).get("b", {}).get("c")
    Returns default if any key is missing or value is not a dict.
    """
    cur = obj
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int):
            cur = cur[key] if key < len(cur) else None
        else:
            return default
        if cur is None:
            return default
    return cur


# ── Initial page extraction ────────────────────────────────────────────────

def extract_innertube_context(html: str, video_id: str):
    """
    Parse the YouTube watch page HTML and extract all data needed
    for the Innertube API: visitor_data, client_version, api_key,
    initial comment continuation token, and video metadata.
    """
    from app.scraper.session import InnertubeContext

    ctx = InnertubeContext(video_id=video_id)

    # ── Extract ytInitialData ─────────────────────────────────────────────
    yt_initial_data = _extract_json_var(html, "ytInitialData")

    # ── Extract ytInitialPlayerResponse ──────────────────────────────────
    player_resp = _extract_json_var(html, "ytInitialPlayerResponse")
    if player_resp:
        vd = deep_get(player_resp, "videoDetails")
        if vd:
            ctx.title        = deep_get(vd, "title")
            ctx.channel_name = deep_get(vd, "author")
            ctx.channel_id   = deep_get(vd, "channelId")
            raw_views        = deep_get(vd, "viewCount")
            ctx.view_count   = _safe_int(raw_views)

    # ── Extract API key ───────────────────────────────────────────────────
    api_key_match = re.search(r'"INNERTUBE_API_KEY"\s*:\s*"([^"]+)"', html)
    if api_key_match:
        ctx.api_key = api_key_match.group(1)

    # ── Extract client version ────────────────────────────────────────────
    version_match = re.search(r'"clientVersion"\s*:\s*"([\d.]+)"', html)
    if version_match:
        ctx.client_version = version_match.group(1)

    # ── Extract visitor_data ──────────────────────────────────────────────
    visitor_match = re.search(r'"visitorData"\s*:\s*"([^"]+)"', html)
    if visitor_match:
        ctx.visitor_data = visitor_match.group(1)

    # ── Extract comment continuation tokens ──────────────────────────────────
    if yt_initial_data:
        ctx.initial_continuation_token = _find_initial_comment_token(yt_initial_data)
        ctx.newest_first_token         = _find_newest_first_token(yt_initial_data)
        # Try to get comment count from initial data
        if not ctx.comment_count:
            ctx.comment_count = _extract_comment_count(yt_initial_data)

    logger.debug(
        "innertube_context_extracted",
        video_id=video_id,
        has_token=bool(ctx.initial_continuation_token),
        client_version=ctx.client_version,
    )
    return ctx


def _extract_json_var(html: str, var_name: str) -> Optional[dict]:
    """Extract a JSON object assigned to a JavaScript variable in page HTML."""
    patterns = [
        rf'var {re.escape(var_name)}\s*=\s*',
        rf'{re.escape(var_name)}\s*=\s*',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if not match:
            continue
        start = html.index('{', match.end())
        try:
            decoder = json.JSONDecoder()
            obj, _  = decoder.raw_decode(html, start)
            return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _find_initial_comment_token(data: dict) -> Optional[str]:
    """
    Extract the initial comment-section continuation token from ytInitialData.

    Tries three paths in priority order:

    1. DIRECT — twoColumnWatchNextResults → itemSectionRenderer with
       a continuationItemRenderer (most reliable, present on most videos).

    2. ENGAGEMENT PANEL — engagementPanels where panelIdentifier ==
       'engagement-panel-comments-section' → subMenuItems[0] token
       (used when the comment section is lazy-loaded into a panel).

    3. FALLBACK — recursive scan for any continuationItemRenderer whose
       token contains 'comments' (case-insensitive URL-decoded).

    NOTE: YouTube no longer puts "COMMENT" in the `request` field of
    continuationCommand — all tokens now use
    CONTINUATION_REQUEST_TYPE_WATCH_NEXT regardless of content type.
    """
    # ── Path 1: twoColumnWatchNextResults ─────────────────────────────────
    contents = deep_get(
        data,
        "contents", "twoColumnWatchNextResults",
        "results", "results", "contents",
    )
    if isinstance(contents, list):
        for item in contents:
            isr = item.get("itemSectionRenderer", {})
            for c in isr.get("contents", []):
                token = deep_get(
                    c,
                    "continuationItemRenderer",
                    "continuationEndpoint",
                    "continuationCommand",
                    "token",
                )
                if token and len(token) > 20:
                    logger.debug("comment_token_found_path1",
                                 token_preview=token[:40])
                    return token

    # ── Path 2: engagement panel 'engagement-panel-comments-section' ──────
    for panel in data.get("engagementPanels", []):
        psr = panel.get("engagementPanelSectionListRenderer", {})
        if psr.get("panelIdentifier") != "engagement-panel-comments-section":
            continue
        # subMenuItems[0] = "Top comments" sort (preferred)
        token = deep_get(
            psr,
            "header",
            "engagementPanelTitleHeaderRenderer",
            "menu",
            "sortFilterSubMenuRenderer",
            "subMenuItems", 0,
            "serviceEndpoint",
            "continuationCommand",
            "token",
        )
        if token and len(token) > 20:
            logger.debug("comment_token_found_path2",
                         token_preview=token[:40])
            return token
        # Also try content → sectionListRenderer
        token = deep_get(
            psr,
            "content", "sectionListRenderer", "contents", 0,
            "itemSectionRenderer", "contents", 0,
            "continuationItemRenderer",
            "continuationEndpoint",
            "continuationCommand",
            "token",
        )
        if token and len(token) > 20:
            logger.debug("comment_token_found_path2b",
                         token_preview=token[:40])
            return token

    # ── Path 3: recursive scan for any token referencing comments ─────────
    token = _recursive_token_search(data, depth=0)
    if token:
        logger.debug("comment_token_found_path3_fallback",
                     token_preview=token[:40])
    return token


def _find_newest_first_token(data: dict) -> Optional[str]:
    """
    Extract the "Newest First" sort continuation token from ytInitialData.

    This token starts a different comment chain that follows ALL comments in
    reverse chronological order — unlike "Top Comments" which YouTube caps at
    a few thousand, this chain can reach the full comment count (e.g. 600k+).

    Location: engagementPanels → engagement-panel-comments-section →
              sortFilterSubMenuRenderer → subMenuItems[1] (index 1 = Newest First)
    """
    for panel in data.get("engagementPanels", []):
        psr = panel.get("engagementPanelSectionListRenderer", {})
        if psr.get("panelIdentifier") != "engagement-panel-comments-section":
            continue
        token = deep_get(
            psr,
            "header",
            "engagementPanelTitleHeaderRenderer",
            "menu",
            "sortFilterSubMenuRenderer",
            "subMenuItems", 1,          # index 1 = "Newest First"
            "serviceEndpoint",
            "continuationCommand",
            "token",
        )
        if token and len(token) > 20:
            logger.debug("newest_first_token_found", token_preview=token[:40])
            return token
    return None


def _recursive_token_search(obj: Any, depth: int) -> Optional[str]:
    """
    Last-resort recursive scan.
    Returns the first long token found inside any continuationItemRenderer.
    (YouTube removed COMMENT-specific request types — all use WATCH_NEXT now.)
    """
    if depth > 15 or obj is None:
        return None
    if isinstance(obj, dict):
        if "continuationItemRenderer" in obj:
            token = deep_get(
                obj,
                "continuationItemRenderer",
                "continuationEndpoint",
                "continuationCommand",
                "token",
            )
            if token and len(token) > 20:
                return token
        for v in obj.values():
            result = _recursive_token_search(v, depth + 1)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _recursive_token_search(item, depth + 1)
            if result:
                return result
    return None


def _extract_comment_count(data: dict) -> Optional[int]:
    """Try to extract the comment count string from ytInitialData."""
    # It's usually in the "engagementPanels" or videoSecondaryInfoRenderer
    count_str = _find_value_by_key(data, "commentsCount", depth_limit=10)
    if count_str:
        return _parse_count_string(str(count_str))
    return None


def _find_value_by_key(obj: Any, key: str, depth_limit: int = 8, _depth: int = 0) -> Any:
    if _depth > depth_limit or obj is None:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_value_by_key(v, key, depth_limit, _depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_value_by_key(item, key, depth_limit, _depth + 1)
            if r is not None:
                return r
    return None


# ── Continuation response parsing ─────────────────────────────────────────

def parse_continuation_response(raw: dict, video_id: str, *, is_reply: bool = False) -> ParsedPage:
    """
    Parse a YouTube /next API response into a ParsedPage.

    YouTube uses TWO formats (supports both):

    ── NEW format (2024+) ────────────────────────────────────────────────────
    Comments are stored as mutations in:
        raw["frameworkUpdates"]["entityBatchUpdate"]["mutations"]
    Each item in continuationItems has a "commentViewModel" with a "commentKey"
    that maps to a "commentEntityPayload" mutation.

    ── OLD format (pre-2024 / some videos) ──────────────────────────────────
    Comments are embedded directly in continuationItems as:
        {"commentThreadRenderer": {"comment": {"commentRenderer": {...}}}}

    Both formats are detected and handled transparently.
    """
    page = ParsedPage()

    endpoints = raw.get("onResponseReceivedEndpoints", [])
    if not endpoints:
        logger.warning("no_response_endpoints", video_id=video_id)
        page.is_last_page = True
        return page

    # Collect all continuation items — can be spread across multiple endpoints
    items = []
    for endpoint in endpoints:
        for action_key in (
            "appendContinuationItemsAction",
            "reloadContinuationItemsCommand",
        ):
            action = endpoint.get(action_key, {})
            items.extend(action.get("continuationItems", []))

    if not items:
        logger.debug("empty_continuation_items", video_id=video_id)
        page.is_last_page = True
        return page

    # Build mutations lookup for new ViewModel format
    mutations_map = _build_mutations_map(raw)

    for item in items:
        # ── Next-page continuation token ─────────────────────────────────
        if "continuationItemRenderer" in item:
            token = deep_get(
                item,
                "continuationItemRenderer",
                "continuationEndpoint",
                "continuationCommand",
                "token",
            )
            if token:
                page.next_token = token
            continue

        # ── Skip non-comment items (header, ads, etc.) ────────────────────
        if "commentsHeaderRenderer" in item:
            continue

        # ── Top-level comment thread ──────────────────────────────────────
        if "commentThreadRenderer" in item:
            thread = item["commentThreadRenderer"]

            # ── NEW format: commentViewModel + mutations ───────────────────
            cvm_outer = thread.get("commentViewModel", {})
            cvm       = cvm_outer.get("commentViewModel", {})   # double-nested

            if cvm and mutations_map:
                comment_key = cvm.get("commentKey")
                if comment_key and comment_key in mutations_map:
                    comment = _parse_comment_from_mutation(
                        comment_key, cvm, mutations_map, video_id, is_reply=False
                    )
                    if comment:
                        page.comments.append(comment)
                        # Reply token — new format path
                        reply_token = deep_get(
                            thread, "replies", "commentRepliesRenderer",
                            "contents", 0,
                            "continuationItemRenderer",
                            "continuationEndpoint",
                            "continuationCommand", "token",
                        )
                        if reply_token and comment.reply_count > 0:
                            page.reply_tokens.append({
                                "comment_id":  comment.comment_id,
                                "video_id":    video_id,
                                "reply_token": reply_token,
                            })
                    continue

            # ── OLD format: comment.commentRenderer ───────────────────────
            comment_renderer = deep_get(thread, "comment", "commentRenderer")
            if comment_renderer:
                comment = _parse_comment_renderer(comment_renderer, video_id, is_reply=False)
                if comment:
                    page.comments.append(comment)
                    reply_token = _extract_reply_token(thread)
                    if reply_token and comment.reply_count > 0:
                        page.reply_tokens.append({
                            "comment_id":  comment.comment_id,
                            "video_id":    video_id,
                            "reply_token": reply_token,
                        })
            continue

        # ── Standalone commentViewModel (reply pages in new format) ───────
        if "commentViewModel" in item:
            cvm_outer = item["commentViewModel"]
            # May be single or double nested
            cvm = cvm_outer.get("commentViewModel", cvm_outer)
            if cvm and mutations_map:
                comment_key = cvm.get("commentKey")
                if comment_key and comment_key in mutations_map:
                    comment = _parse_comment_from_mutation(
                        comment_key, cvm, mutations_map, video_id, is_reply=True
                    )
                    if comment:
                        page.comments.append(comment)
            continue

        # ── Reply item — old format ───────────────────────────────────────
        if "commentRenderer" in item:
            comment = _parse_comment_renderer(
                item["commentRenderer"], video_id, is_reply=True
            )
            if comment:
                page.comments.append(comment)
            continue

    page.is_last_page = page.next_token is None

    logger.debug(
        "page_parsed",
        video_id      = video_id,
        is_reply      = is_reply,
        comments      = len(page.comments),
        reply_tokens  = len(page.reply_tokens),
        has_next      = bool(page.next_token),
        is_last       = page.is_last_page,
        used_mutations = bool(mutations_map),
    )

    return page


# ── New ViewModel + mutations helpers ─────────────────────────────────────

def _build_mutations_map(raw: dict) -> dict:
    """
    Build {entityKey: payload_dict} from frameworkUpdates mutations.

    YouTube 2024+ format stores comment data outside continuationItems,
    in a parallel 'frameworkUpdates.entityBatchUpdate.mutations' list.
    Each mutation has an 'entityKey' and a 'payload' with typed sub-dicts.
    """
    result: dict = {}
    mutations = deep_get(
        raw, "frameworkUpdates", "entityBatchUpdate", "mutations"
    )
    if not isinstance(mutations, list):
        return result
    for m in mutations:
        key = m.get("entityKey")
        payload = m.get("payload")
        if key and payload:
            result[key] = payload
    return result


def _parse_comment_from_mutation(
    comment_key: str,
    cvm:          dict,
    mutations_map: dict,
    video_id:     str,
    *,
    is_reply: bool,
) -> Optional["CommentDocument"]:
    """
    Parse a comment from the new ViewModel + mutations format.

    comment_key → mutations_map[comment_key]["commentEntityPayload"]
                  contains properties, author, toolbar
    cvm.toolbarStateKey → mutations_map[toolbarStateKey]["engagementToolbarStateEntityPayload"]
                          contains heartState
    cvm.pinnedText      → non-empty when comment is pinned
    """
    payload = mutations_map.get(comment_key, {})
    cep     = payload.get("commentEntityPayload")
    if not cep:
        return None

    props   = cep.get("properties", {})
    author  = cep.get("author", {})
    toolbar = cep.get("toolbar", {})

    # ── Identity ─────────────────────────────────────────────────────────
    comment_id = props.get("commentId") or cvm.get("commentId")
    if not comment_id:
        return None

    # ── Text ─────────────────────────────────────────────────────────────
    # New format uses a plain string, not the old runs[] array
    text = deep_get(props, "content", "content") or ""
    if not text:
        return None

    # ── Author ───────────────────────────────────────────────────────────
    author_name       = author.get("displayName")
    author_channel_id = author.get("channelId")
    is_channel_owner  = bool(author.get("isCreator", False))
    is_member         = False   # no direct signal in new format

    # ── Engagement ───────────────────────────────────────────────────────
    like_str   = (toolbar.get("likeCountNotliked")
                  or toolbar.get("likeCountLiked")
                  or "0")
    like_count = _parse_count_string(like_str)
    like_exact = like_str.isdigit()   # "123" is exact; "6.5K" is not

    reply_str   = toolbar.get("replyCount") or "0"
    reply_count = _safe_int(reply_str) if reply_str.isdigit() else _parse_count_string(reply_str)
    reply_count = reply_count or 0

    # ── Heart state ───────────────────────────────────────────────────────
    toolbar_state_key = props.get("toolbarStateKey") or cvm.get("toolbarStateKey")
    is_hearted        = False
    if toolbar_state_key:
        ts_payload = mutations_map.get(toolbar_state_key, {})
        ts = ts_payload.get("engagementToolbarStateEntityPayload", {})
        is_hearted = ts.get("heartState") == "TOOLBAR_HEART_STATE_HEARTED"

    # ── Pinned ────────────────────────────────────────────────────────────
    is_pinned = bool(cvm.get("pinnedText"))

    # ── Timestamps ───────────────────────────────────────────────────────
    pub_text              = props.get("publishedTime") or ""
    is_edited, pub_clean  = _parse_edited_flag(pub_text)
    pub_approx, precision = _approximate_datetime(pub_clean)

    # ── Reply parentage ───────────────────────────────────────────────────
    reply_level = props.get("replyLevel", 0)
    if reply_level > 0:
        is_reply = True
    parent_id = None
    if is_reply and "." in comment_id:
        parent_id = comment_id.split(".")[0]

    reply_to_id, reply_link_type = _heuristic_reply_to(text, is_reply)

    # ── Build document ────────────────────────────────────────────────────
    try:
        doc = CommentDocument(
            comment_id               = comment_id,
            video_id                 = video_id,
            is_reply                 = is_reply,
            parent_comment_id        = parent_id,
            reply_to_comment_id      = reply_to_id,
            reply_link_type          = reply_link_type,
            thread_depth             = 1 if is_reply else 0,
            text_formatted           = [{"text": text}],   # plain string → minimal run
            is_edited                = is_edited,
            author_name              = author_name,
            author_channel_id        = author_channel_id,
            author_is_channel_owner  = is_channel_owner,
            author_is_member         = is_member,
            like_count               = like_count,
            like_count_display       = like_str,
            like_count_exact         = like_exact,
            reply_count              = reply_count,
            is_pinned                = is_pinned,
            is_hearted               = is_hearted,
            comment_type             = CommentType.STANDARD,
            super_thanks_amount      = None,
            published_time_text      = pub_clean,
            published_at_approx      = pub_approx,
            published_at_precision   = precision,
            status                   = CommentStatus.ACTIVE,
        )
        doc.set_text(text)
        return doc

    except Exception as exc:
        logger.warning(
            "comment_parse_error_new_format",
            comment_id = comment_id,
            error      = str(exc),
        )
        return None


def _parse_comment_renderer(
    cr: dict, video_id: str, *, is_reply: bool
) -> Optional[CommentDocument]:
    """
    Parse a single commentRenderer dict into a CommentDocument.
    Returns None if the essential comment_id or text is missing.
    """
    try:
        comment_id = cr.get("commentId")
        if not comment_id:
            return None

        # ── Text ────────────────────────────────────────────────────────
        content = cr.get("contentText", {})
        text, text_formatted = _parse_runs(content)
        if not text:
            return None

        # ── Author ──────────────────────────────────────────────────────
        author_name       = _runs_to_text(cr.get("authorText", {}))
        author_channel_id = deep_get(cr, "authorEndpoint", "browseEndpoint", "browseId")

        # ── Engagement ──────────────────────────────────────────────────
        like_count, like_display, like_exact = _parse_like_count(cr)
        reply_count = _safe_int(cr.get("replyCount")) or 0

        # ── Flags ────────────────────────────────────────────────────────
        is_pinned          = "pinnedCommentBadge" in cr and cr["pinnedCommentBadge"] is not None
        is_hearted         = "creatorHeart" in cr and cr["creatorHeart"] is not None
        is_channel_owner   = bool(cr.get("authorIsChannelOwner", False))
        is_member          = "memberBadge" in cr and cr["memberBadge"] is not None

        # ── Timestamps ──────────────────────────────────────────────────
        pub_text = _runs_to_text(cr.get("publishedTimeText", {}))
        is_edited, pub_text_clean = _parse_edited_flag(pub_text)
        pub_approx, precision     = _approximate_datetime(pub_text_clean)

        # ── Comment type ─────────────────────────────────────────────────
        comment_type, super_thanks_amount = _parse_comment_type(cr)

        # ── Parent (for replies) ─────────────────────────────────────────
        parent_id = None
        if is_reply:
            # Reply comment_id format: "parent_id.reply_id"
            if "." in comment_id:
                parent_id = comment_id.split(".")[0]

        # ── Reply-to heuristic ───────────────────────────────────────────
        reply_to_id, reply_link_type = _heuristic_reply_to(text, is_reply)

        # ── Build document ───────────────────────────────────────────────
        doc              = CommentDocument(
            comment_id               = comment_id,
            video_id                 = video_id,
            is_reply                 = is_reply,
            parent_comment_id        = parent_id,
            reply_to_comment_id      = reply_to_id,
            reply_link_type          = reply_link_type,
            thread_depth             = 1 if is_reply else 0,
            text_formatted           = text_formatted,
            is_edited                = is_edited,
            author_name              = author_name,
            author_channel_id        = author_channel_id,
            author_is_channel_owner  = is_channel_owner,
            author_is_member         = is_member,
            like_count               = like_count,
            like_count_display       = like_display,
            like_count_exact         = like_exact,
            reply_count              = reply_count,
            is_pinned                = is_pinned,
            is_hearted               = is_hearted,
            comment_type             = comment_type,
            super_thanks_amount      = super_thanks_amount,
            published_time_text      = pub_text_clean,
            published_at_approx      = pub_approx,
            published_at_precision   = precision,
            status                   = CommentStatus.ACTIVE,
        )
        doc.set_text(text)   # sets text + computes text_hash
        return doc

    except Exception as exc:
        logger.warning(
            "comment_parse_error",
            comment_id=cr.get("commentId", "unknown"),
            error=str(exc),
        )
        return None


# ── Field parsers ──────────────────────────────────────────────────────────

def _parse_runs(content: dict) -> tuple[str, list]:
    """Convert YouTube's runs array to (plain_text, formatted_runs_list)."""
    runs = content.get("runs", [])
    parts = []
    for run in runs:
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            # Extract emoji shortcut or unicode
            shortcuts = deep_get(run, "emoji", "shortcuts")
            if shortcuts:
                parts.append(shortcuts[0])
            else:
                emoji_id = deep_get(run, "emoji", "emojiId", default="")
                parts.append(emoji_id)
    return "".join(parts), runs


def _runs_to_text(content: dict) -> Optional[str]:
    text, _ = _parse_runs(content)
    return text or None


def _parse_like_count(cr: dict) -> tuple[int, str, bool]:
    """
    Returns (count_int, display_str, is_exact).
    YouTube returns like_count as an int for small counts and may
    include an accessibility string for large counts.
    """
    raw = cr.get("likeCount")

    # Case 1: integer directly
    if isinstance(raw, int):
        return raw, str(raw), True

    # Case 2: string in accessibility data  ("1,247 likes")
    acc_label = deep_get(
        cr, "actionButtons", "commentActionButtonsRenderer",
        "likeButton", "toggleButtonRenderer",
        "accessibilityData", "accessibilityData", "label",
    )
    if acc_label:
        numbers = re.findall(r"[\d,]+", acc_label)
        if numbers:
            count = _safe_int(numbers[0].replace(",", ""))
            if count is not None:
                return count, str(count), True

    # Case 3: abbreviated string ("1.2K")
    if isinstance(raw, str) and raw:
        count = _parse_count_string(raw)
        return count, raw, False

    return 0, "0", True


def _parse_count_string(s: str) -> int:
    """Parse "1.2K" → 1200, "3.4M" → 3400000, "12" → 12."""
    s = s.strip().replace(",", "")
    try:
        if s.endswith("K") or s.endswith("k"):
            return int(float(s[:-1]) * 1_000)
        if s.endswith("M") or s.endswith("m"):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith("B") or s.endswith("b"):
            return int(float(s[:-1]) * 1_000_000_000)
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _parse_edited_flag(pub_text: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Detect "(edited)" in the published time text.
    Returns (is_edited, cleaned_pub_text).
    """
    if not pub_text:
        return False, pub_text
    if "(edited)" in pub_text.lower():
        clean = re.sub(r"\s*\(edited\)\s*", "", pub_text, flags=re.IGNORECASE).strip()
        return True, clean or pub_text
    return False, pub_text


def _approximate_datetime(
    text: Optional[str],
) -> tuple[Optional[datetime], str]:
    """
    Convert "2 years ago", "3 months ago", etc. to an approximate datetime.
    Returns (datetime, precision_label).
    """
    if not text:
        return None, TimestampPrecision.APPROXIMATE

    now = datetime.now(timezone.utc)
    text = text.lower().strip()

    patterns = [
        (r"(\d+)\s+second",  1,           TimestampPrecision.EXACT),
        (r"(\d+)\s+minute",  60,          TimestampPrecision.EXACT),
        (r"(\d+)\s+hour",    3600,        TimestampPrecision.DAY),
        (r"(\d+)\s+day",     86400,       TimestampPrecision.DAY),
        (r"(\d+)\s+week",    604800,      TimestampPrecision.WEEK),
        (r"(\d+)\s+month",   2592000,     TimestampPrecision.MONTH),
        (r"(\d+)\s+year",    31536000,    TimestampPrecision.YEAR),
    ]
    for pattern, multiplier, precision in patterns:
        m = re.search(pattern, text)
        if m:
            n       = int(m.group(1))
            seconds = n * multiplier
            from datetime import timedelta
            approx  = now - timedelta(seconds=seconds)
            return approx, precision

    return None, TimestampPrecision.APPROXIMATE


def _parse_comment_type(cr: dict) -> tuple[str, Optional[str]]:
    """Detect Super Thanks and Members-only comment types."""
    # Super Thanks
    chip = deep_get(cr, "superThanksChip", "superThanksChipRenderer")
    if chip:
        amount = deep_get(chip, "amount", "simpleText") or deep_get(chip, "amount")
        return CommentType.SUPER_THANKS, str(amount) if amount else None

    # Paid comment chip (alternative path)
    paid = cr.get("paidCommentChipRenderer")
    if paid:
        amount = deep_get(paid, "pdgCommentChipRenderer", "chipText", "simpleText")
        return CommentType.SUPER_THANKS, str(amount) if amount else None

    # Members-only
    if "memberBadge" in cr and cr.get("memberBadge"):
        return CommentType.MEMBERS_ONLY, None

    return CommentType.STANDARD, None


def _extract_reply_token(thread: dict) -> Optional[str]:
    """Extract the reply continuation token from a commentThreadRenderer."""
    # Path 1: replies.commentRepliesRenderer.continuations[0].nextContinuationData.continuation
    token = deep_get(
        thread, "replies", "commentRepliesRenderer",
        "continuations", 0, "nextContinuationData", "continuation",
    )
    if token:
        return token
    # Path 2: contents[0].continuationItemRenderer...
    token = deep_get(
        thread, "replies", "commentRepliesRenderer",
        "contents", 0, "continuationItemRenderer",
        "continuationEndpoint", "continuationCommand", "token",
    )
    return token


def _heuristic_reply_to(text: str, is_reply: bool) -> tuple[Optional[str], str]:
    """
    Attempt to detect which reply this comment is responding to via @mention.
    Returns (None, "unknown") if we cannot determine it confidently.
    NOTE: This returns a channel_id or None — NOT a comment_id.
    The caller links it to a comment_id in the pipeline if needed.
    """
    if not is_reply:
        return None, ReplyLinkType.API
    # Look for @username at the very start of the text
    m = re.match(r"^@([^\s,]+)", text.strip())
    if m:
        # We found a mention but we can't resolve it to a comment_id here
        # — return the mention text and flag as heuristic
        return None, ReplyLinkType.HEURISTIC
    return None, ReplyLinkType.UNKNOWN


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
