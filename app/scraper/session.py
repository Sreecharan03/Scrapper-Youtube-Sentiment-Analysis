"""
app/scraper/session.py
=======================
aiohttp ClientSession factory and YouTube visitor context management.

A "scraper session" is a single aiohttp.ClientSession that holds:
  • Connection pool (reused across all requests in a batch)
  • Cookies accumulated from the initial page load
  • The innertube context (client version, visitor_data) needed by the API

WHY NOT A SINGLETON:
  Celery tasks are separate processes.  Each task creates its own session
  at startup and closes it on completion.  This avoids sharing state
  across workers and lets each task have independent cookie jars.

USAGE:
  async with ScraperSession(video_id) as session:
      context = await session.initialise()   # fetch page, extract tokens
      raw = await session.post_continuation(token)
"""

import asyncio
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from app.core.logging import get_logger
from app.scraper.constants import (
    CONNECT_TIMEOUT, DEFAULT_HEADERS, INNERTUBE_POST_HEADERS,
    READ_TIMEOUT, REQUEST_DELAY_MAX_MS, REQUEST_DELAY_MIN_MS,
    YT_API_KEY_FALLBACK, YT_CLIENT_VERSION_FALLBACK,
    YT_INNERTUBE_URL, YT_WATCH_URL,
)

logger = get_logger(__name__)


@dataclass
class InnertubeContext:
    """All data extracted from the initial page load required for API calls."""
    video_id:       str
    api_key:        str   = YT_API_KEY_FALLBACK
    client_version: str   = YT_CLIENT_VERSION_FALLBACK
    visitor_data:   str   = ""
    initial_continuation_token: Optional[str] = None   # "Top Comments" sort
    newest_first_token:         Optional[str] = None   # "Newest First" sort — gives ALL comments
    # Video metadata extracted from ytInitialPlayerResponse
    title:          Optional[str] = None
    channel_name:   Optional[str] = None
    channel_id:     Optional[str] = None
    view_count:     Optional[int] = None
    comment_count:  Optional[int] = None


class ScraperSession:
    """
    Manages one aiohttp session for the duration of a single Celery task.

    Usage:
        async with ScraperSession(video_id) as scraper:
            ctx = await scraper.initialise()
            raw_page = await scraper.post_continuation(ctx, token)
    """

    def __init__(self, video_id: str) -> None:
        self.video_id = video_id
        self._session: Optional[aiohttp.ClientSession] = None
        self.context:  Optional[InnertubeContext]      = None

    async def __aenter__(self) -> "ScraperSession":
        timeout = aiohttp.ClientTimeout(
            connect=CONNECT_TIMEOUT,
            total=READ_TIMEOUT + CONNECT_TIMEOUT,
        )
        self._session = aiohttp.ClientSession(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=10,           # max concurrent connections per session
                ssl=True,
                enable_cleanup_closed=True,
            ),
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            # Brief pause for SSL connection teardown (reduced from 0.25s)
            await asyncio.sleep(0.05)

    async def initialise(self) -> InnertubeContext:
        """
        Fetch the YouTube watch page, extract InnertubeContext.
        Must be called before post_continuation().
        """
        url  = YT_WATCH_URL.format(video_id=self.video_id)
        html = await self._get_html(url)

        from app.scraper.parser import extract_innertube_context
        ctx = extract_innertube_context(html, self.video_id)
        self.context = ctx

        logger.info(
            "scraper_session_initialised",
            video_id=self.video_id,
            client_version=ctx.client_version,
            has_initial_token=bool(ctx.initial_continuation_token),
            comment_count=ctx.comment_count,
        )
        return ctx

    async def post_continuation(
        self, token: str, context: Optional[InnertubeContext] = None
    ) -> dict:
        """
        POST to YouTube's Innertube /next endpoint with a continuation token.
        Returns the raw JSON response dict.

        Raises:
            aiohttp.ClientResponseError — on HTTP errors (caller handles)
        """
        ctx = context or self.context
        if ctx is None:
            raise RuntimeError("Call initialise() before post_continuation()")

        body = {
            "context": {
                "client": {
                    "clientName":    "WEB",
                    "clientVersion": ctx.client_version,
                    "hl":            "en",
                    "gl":            "US",
                    **({"visitorData": ctx.visitor_data} if ctx.visitor_data else {}),
                }
            },
            "continuation": token,
        }

        url = f"{YT_INNERTUBE_URL}?key={ctx.api_key}&prettyPrint=false"
        headers = {
            **INNERTUBE_POST_HEADERS,
            "X-YouTube-Client-Version": ctx.client_version,
            "Referer": YT_WATCH_URL.format(video_id=self.video_id),
        }

        if self._session is None:
            raise RuntimeError("ScraperSession not entered — use async with")

        async with self._session.post(url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def _get_html(self, url: str) -> str:
        if self._session is None:
            raise RuntimeError("ScraperSession not entered — use async with")
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def random_delay(self) -> None:
        """Human-like jitter between API calls."""
        delay = random.randint(REQUEST_DELAY_MIN_MS, REQUEST_DELAY_MAX_MS) / 1000.0
        await asyncio.sleep(delay)
