"""
app/core/cache.py
==================
Structured cache and session manager built on top of Redis.

SUBSYSTEMS:
  1. Core get/set/delete/exists with JSON serialization
  2. Distributed job lock    — atomic SETNX prevents duplicate scrapes
  3. Sliding-window rate limiter — INCR + EXPIRE per time bucket
  4. Video metadata cache    — warm cache in front of MongoDB
  5. Scraper session store   — continuation tokens, state between pages
  6. Reply queue             — Redis LIST fed by TLC workers, drained by reply workers
  7. Reply pending counter   — atomic DECR tracks when all replies are done
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis_client import get_redis

logger = get_logger(__name__)

KEY_VERSION = "v1"


class TTL:
    JOB_LOCK          = 60 * 60 * 8   # 8 hours — longer than max scrape duration
    VIDEO_METADATA    = 60 * 60 * 6   # 6 hours
    SCRAPER_SESSION   = 60 * 60 * 4   # 4 hours
    RATE_LIMIT_WINDOW = 60 * 60       # 1 hour
    SHORT             = 60 * 5        # 5 minutes
    REPLY_QUEUE       = 60 * 60 * 12  # 12 hours — reply queue persists across restarts


class CacheManager:
    """
    Structured cache and session manager.
    Stateless — create anywhere or use the module-level `cache` singleton.

    FASTAPI USAGE (module-level singleton):
        from app.core.cache import cache
        # The singleton uses the FastAPI-managed Redis connection (get_redis()).

    CELERY USAGE (inject a task-scoped Redis client):
        from app.scraper.pipeline import make_redis_client
        redis_client = make_redis_client()
        cache = CacheManager(redis_client=redis_client)
        # Close the client after asyncio.run() completes.
    """

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self._settings    = get_settings()
        self._redis_client = redis_client   # injected (Celery) or None (FastAPI)

    def _get_redis(self) -> Any:
        """
        Return the Redis client to use.
        In FastAPI: the global singleton managed by the lifespan.
        In Celery:  the injected client created per-task.
        """
        if self._redis_client is not None:
            return self._redis_client
        return get_redis()   # FastAPI singleton

    # ── Key builder ────────────────────────────────────────────────────────

    def _key(self, namespace: str, *parts: str) -> str:
        app = self._settings.app_name.replace("-", "_")
        env = self._settings.app_env[:3]
        return ":".join([app, env, KEY_VERSION, namespace] + list(parts))

    # ── Core primitives ────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        return await self._get_redis().get(key)

    async def set(self, key: str, value: str, *, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds:
            await self._get_redis().setex(key, ttl_seconds, value)
        else:
            await self._get_redis().set(key, value)

    async def delete(self, *keys: str) -> int:
        return await self._get_redis().delete(*keys)

    async def exists(self, key: str) -> bool:
        return bool(await self._get_redis().exists(key))

    async def ttl(self, key: str) -> int:
        return await self._get_redis().ttl(key)

    async def get_json(self, key: str) -> Optional[Any]:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("cache_json_decode_error", key=key)
            return None

    async def set_json(self, key: str, value: Any, *, ttl_seconds: Optional[int] = None) -> None:
        await self.set(key, json.dumps(value, default=str), ttl_seconds=ttl_seconds)

    # ── Distributed Job Lock ───────────────────────────────────────────────

    async def acquire_job_lock(self, video_id: str, *, ttl_seconds: int = TTL.JOB_LOCK) -> bool:
        """Atomic SETNX — only one caller wins. Returns True if lock acquired."""
        key = self._key("job_lock", video_id)
        acquired = await self._get_redis().set(key, "1", nx=True, ex=ttl_seconds)
        result = acquired is not None
        logger.debug("job_lock_attempt", video_id=video_id, acquired=result)
        return result

    async def release_job_lock(self, video_id: str) -> None:
        await self.delete(self._key("job_lock", video_id))

    async def is_job_locked(self, video_id: str) -> bool:
        return await self.exists(self._key("job_lock", video_id))

    # ── Rate Limiter ───────────────────────────────────────────────────────

    async def increment_rate_limit(
        self, namespace: str, *, window_seconds: int = TTL.RATE_LIMIT_WINDOW
    ) -> int:
        now    = datetime.now(timezone.utc)
        bucket = int(now.timestamp() // window_seconds) * window_seconds
        key    = self._key("rate_limit", namespace, str(bucket))
        pipe   = self._get_redis().pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds * 2)
        results = await pipe.execute()
        count   = results[0]
        logger.debug("rate_limit_inc", namespace=namespace, count=count)
        return count

    async def get_rate_limit_count(
        self, namespace: str, *, window_seconds: int = TTL.RATE_LIMIT_WINDOW
    ) -> int:
        now    = datetime.now(timezone.utc)
        bucket = int(now.timestamp() // window_seconds) * window_seconds
        raw    = await self.get(self._key("rate_limit", namespace, str(bucket)))
        return int(raw) if raw else 0

    async def reset_rate_limit(self, namespace: str) -> None:
        now    = datetime.now(timezone.utc)
        bucket = int(now.timestamp() // TTL.RATE_LIMIT_WINDOW) * TTL.RATE_LIMIT_WINDOW
        await self.delete(self._key("rate_limit", namespace, str(bucket)))

    # ── Video Metadata Cache ───────────────────────────────────────────────

    async def get_video_metadata(self, video_id: str) -> Optional[dict]:
        data = await self.get_json(self._key("video_meta", video_id))
        if data:
            logger.debug("video_meta_cache_hit", video_id=video_id)
        return data

    async def set_video_metadata(
        self, video_id: str, metadata: dict, *, ttl_seconds: int = TTL.VIDEO_METADATA
    ) -> None:
        await self.set_json(self._key("video_meta", video_id), metadata, ttl_seconds=ttl_seconds)

    async def invalidate_video_metadata(self, video_id: str) -> None:
        await self.delete(self._key("video_meta", video_id))

    # ── Scraper Session Store ──────────────────────────────────────────────

    async def get_scraper_session(self, job_id: str) -> Optional[dict]:
        return await self.get_json(self._key("scraper_session", job_id))

    async def set_scraper_session(
        self, job_id: str, state: dict, *, ttl_seconds: int = TTL.SCRAPER_SESSION
    ) -> None:
        await self.set_json(self._key("scraper_session", job_id), state, ttl_seconds=ttl_seconds)
        logger.debug("scraper_session_saved", job_id=job_id,
                     page=state.get("sub_batch_number", 0))

    async def update_tlc_token(
        self,
        job_id: str,
        token: str,
        sub_batch_number: int,
        comments_written: int,
        batch_number: int,
    ) -> None:
        """Fast in-place update of the hot token — called after every sub-batch."""
        existing = await self.get_scraper_session(job_id) or {}
        existing.update({
            "current_tlc_token":      token,
            "sub_batch_number":       sub_batch_number,
            "comments_written_total": comments_written,
            "current_batch_number":   batch_number,
            "last_updated":           datetime.now(timezone.utc).isoformat(),
        })
        await self.set_scraper_session(job_id, existing)

    async def clear_scraper_session(self, job_id: str) -> None:
        await self.delete(self._key("scraper_session", job_id))

    # ── Reply Queue (Redis LIST) ───────────────────────────────────────────
    #
    # The TLC batch worker pushes reply tokens here.
    # The reply worker pool pops from here continuously.
    # Using a Redis LIST with RPUSH/BLPOP gives us:
    #   • Atomicity — each token popped by exactly one worker
    #   • Persistence — survives worker restarts (Redis persists the list)
    #   • Blocking pop — reply workers wait efficiently without polling

    def _reply_queue_key(self, job_id: str) -> str:
        return self._key("reply_queue", job_id)

    def _reply_pending_key(self, job_id: str) -> str:
        return self._key("reply_pending", job_id)

    async def push_reply_tokens(
        self, job_id: str, tokens: list[dict]
    ) -> int:
        """
        Push a batch of reply tokens onto the queue.
        Each token dict must have: {comment_id, video_id, reply_token}

        Returns total queue length after push.
        """
        if not tokens:
            return 0
        pipe       = self._get_redis().pipeline()
        queue_key  = self._reply_queue_key(job_id)
        pending_key = self._reply_pending_key(job_id)
        for t in tokens:
            pipe.rpush(queue_key, json.dumps(t))
        pipe.incrby(pending_key, len(tokens))
        pipe.expire(queue_key,   TTL.REPLY_QUEUE)
        pipe.expire(pending_key, TTL.REPLY_QUEUE)
        results = await pipe.execute()
        logger.debug("reply_tokens_pushed", job_id=job_id, count=len(tokens))
        return results[0]  # queue length after push

    async def pop_reply_token(self, job_id: str, timeout: int = 5) -> Optional[dict]:
        """
        Non-blocking pop from the reply queue.
        Returns None if queue is empty.
        """
        key    = self._reply_queue_key(job_id)
        result = await self._get_redis().lpop(key)
        if result is None:
            return None
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error("reply_token_parse_error", raw=result)
            return None

    async def reply_queue_length(self, job_id: str) -> int:
        """Return how many tokens are waiting in the queue."""
        return await self._get_redis().llen(self._reply_queue_key(job_id))

    async def decrement_reply_pending(self, job_id: str) -> int:
        """
        Decrement the pending counter when a reply batch completes.
        Returns the new counter value.
        """
        key   = self._reply_pending_key(job_id)
        value = await self._get_redis().decr(key)
        logger.debug("reply_pending_decremented", job_id=job_id, remaining=value)
        return max(0, value)  # never go below 0

    async def get_reply_pending_count(self, job_id: str) -> int:
        raw = await self.get(self._reply_pending_key(job_id))
        return max(0, int(raw)) if raw else 0

    async def clear_reply_queue(self, job_id: str) -> None:
        """Remove reply queue + counter on job completion."""
        await self.delete(
            self._reply_queue_key(job_id),
            self._reply_pending_key(job_id),
        )

    # ── Health check ──────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            return await self._get_redis().ping() is True
        except Exception as exc:
            logger.error("redis_ping_failed", error=str(exc))
            return False


# Module-level singleton — import this everywhere
cache = CacheManager()
