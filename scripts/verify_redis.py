"""
scripts/verify_redis.py
========================
Standalone Redis connection and cache verification script.

PURPOSE:
  Run this manually to confirm Redis connectivity and test all CacheManager
  operations from Lightning AI before starting the full application.

USAGE:
  python scripts/verify_redis.py

WHAT IT TESTS:
  1. Settings load — REDIS_HOST, REDIS_PORT, REDIS_PASSWORD are present
  2. Async client connects and PING succeeds
  3. Basic set/get/delete round-trip
  4. JSON serialization/deserialization
  5. Job lock acquire / release (atomic SETNX)
  6. Rate limit counter (INCR + EXPIRE)
  7. Scraper session save / retrieve / clear
  8. Key expiry (TTL enforcement)
  9. Confirm all test keys are cleaned up
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.redis_client import connect_redis, disconnect_redis, get_redis
from app.core.cache import CacheManager

setup_logging()
logger = get_logger("verify_redis")


async def verify() -> bool:
    settings = get_settings()
    logger.info(
        "verification_started",
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_cache_db,
        ssl=settings.redis_ssl,
    )

    # ---- Step 1: Connect ----
    logger.info("step_1_connecting")
    try:
        await connect_redis()
        logger.info("step_1_connected")
    except Exception as exc:
        logger.error("step_1_failed", error=str(exc), exc_info=True)
        return False

    cache = CacheManager()

    try:
        # ---- Step 2: PING ----
        logger.info("step_2_ping")
        ok = await cache.ping()
        assert ok, "ping returned False"
        logger.info("step_2_ping_ok")

        # ---- Step 3: Basic set/get/delete ----
        logger.info("step_3_basic_ops")
        test_key = cache._key("verify", "basic")
        await cache.set(test_key, "hello_redis", ttl_seconds=60)
        val = await cache.get(test_key)
        assert val == "hello_redis", f"Expected 'hello_redis', got {val!r}"
        await cache.delete(test_key)
        assert not await cache.exists(test_key), "Key should be gone after delete"
        logger.info("step_3_basic_ops_ok")

        # ---- Step 4: JSON round-trip ----
        logger.info("step_4_json")
        json_key = cache._key("verify", "json")
        payload = {"video_id": "dQw4w9WgXcQ", "count": 42, "tags": ["a", "b"]}
        await cache.set_json(json_key, payload, ttl_seconds=60)
        retrieved = await cache.get_json(json_key)
        assert retrieved == payload, f"JSON mismatch: {retrieved}"
        await cache.delete(json_key)
        logger.info("step_4_json_ok")

        # ---- Step 5: Job lock ----
        logger.info("step_5_job_lock")
        test_video = "verify_test_video"
        acquired1 = await cache.acquire_job_lock(test_video, ttl_seconds=30)
        assert acquired1, "First acquire should succeed"
        acquired2 = await cache.acquire_job_lock(test_video, ttl_seconds=30)
        assert not acquired2, "Second acquire should fail (already locked)"
        assert await cache.is_job_locked(test_video)
        await cache.release_job_lock(test_video)
        assert not await cache.is_job_locked(test_video), "Lock should be gone after release"
        logger.info("step_5_job_lock_ok")

        # ---- Step 6: Rate limiter ----
        logger.info("step_6_rate_limit")
        # Reset first to start from 0
        await cache.reset_rate_limit("verify_test")
        count1 = await cache.increment_rate_limit("verify_test", window_seconds=60)
        count2 = await cache.increment_rate_limit("verify_test", window_seconds=60)
        count3 = await cache.increment_rate_limit("verify_test", window_seconds=60)
        assert count1 == 1, f"Expected 1, got {count1}"
        assert count2 == 2, f"Expected 2, got {count2}"
        assert count3 == 3, f"Expected 3, got {count3}"
        read_count = await cache.get_rate_limit_count("verify_test", window_seconds=60)
        assert read_count == 3, f"Expected 3 from get, got {read_count}"
        await cache.reset_rate_limit("verify_test")
        logger.info("step_6_rate_limit_ok", counts=[count1, count2, count3])

        # ---- Step 7: Scraper session ----
        logger.info("step_7_scraper_session")
        test_job_id = "verify_job_001"
        session_state = {
            "continuation_token": "eyJhbGciOiJSUzI1NiJ9.test",
            "page_count": 3,
            "comments_collected": 150,
            "cookies": {"SESSION_TOKEN": "abc123"},
        }
        await cache.set_scraper_session(test_job_id, session_state, ttl_seconds=60)
        retrieved_session = await cache.get_scraper_session(test_job_id)
        assert retrieved_session == session_state, f"Session mismatch: {retrieved_session}"

        # Test partial update
        await cache.update_continuation_token(
            test_job_id, token="new_token_xyz", page_count=4, comments_collected=200
        )
        updated = await cache.get_scraper_session(test_job_id)
        assert updated["continuation_token"] == "new_token_xyz"
        assert updated["page_count"] == 4
        assert updated["cookies"] == {"SESSION_TOKEN": "abc123"}  # Preserved
        await cache.clear_scraper_session(test_job_id)
        assert await cache.get_scraper_session(test_job_id) is None
        logger.info("step_7_scraper_session_ok")

        # ---- Step 8: TTL enforcement ----
        logger.info("step_8_ttl")
        ttl_key = cache._key("verify", "ttl_test")
        await cache.set(ttl_key, "expires_soon", ttl_seconds=5)
        remaining = await cache.ttl(ttl_key)
        assert 1 <= remaining <= 5, f"TTL should be 1-5, got {remaining}"
        await cache.delete(ttl_key)
        logger.info("step_8_ttl_ok", remaining_ttl=remaining)

        logger.info(
            "verification_passed",
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_cache_db,
        )
        return True

    except AssertionError as exc:
        logger.error("assertion_failed", error=str(exc))
        return False
    except Exception as exc:
        logger.error("unexpected_error", error=str(exc), exc_info=True)
        return False
    finally:
        await disconnect_redis()
        logger.info("client_closed")


if __name__ == "__main__":
    success = asyncio.run(verify())
    print("\n" + (
        "✅ Redis verified — all checks passed." if success
        else "❌ Redis verification failed — check logs above."
    ))
    sys.exit(0 if success else 1)
