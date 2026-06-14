"""
app/workers/tasks/classification_tasks.py
==========================================
Celery task: classify all comments for a video using Groq llama-3.1-8b-instant.

FLOW:
  1. Guard: summary must exist and be completed
  2. Mark comment_analysis doc as processing
  3. Load all (or only failed) comments from MongoDB
  4. Build parent-text map for replies
  5. Run async classification (CommentClassifier) with preprocessing pipeline
  6. Bulk-update all comment docs with labels + sentiment
  7. Compute aggregate stats and store in comment_analysis
  8. Mark as completed
  9. Auto-retry: if failed_count > 0 and auto_retry_count < MAX_AUTO_RETRIES,
     self-chain another retry task automatically

AUTO-RETRY LOGIC:
  MAX_AUTO_RETRIES = 3
  Each retry only processes comments with classification_status="failed".
  Retries are self-chaining — no manual intervention needed.
  After MAX_AUTO_RETRIES the remaining failures are accepted as permanent.
"""

import asyncio
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories.classification_repo import ClassificationRepository, ClassificationStatus
from app.db.repositories.comment_repo import CommentRepository
from app.scraper.pipeline import make_db_client
from app.services.classifier import CommentClassifier
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

MAX_AUTO_RETRIES = 3   # max automatic retry rounds after the initial classification


@celery_app.task(
    bind                = True,
    name                = "classify_comments",
    queue               = "scraper",
    max_retries         = 1,
    default_retry_delay = 60,
    soft_time_limit     = 1800,
    time_limit          = 1860,
    acks_late           = True,
    ignore_result       = True,
)
def classify_comments(
    self,
    *,
    video_id:         str,
    retry_failed:     bool = False,
    auto_retry_count: int  = 0,
) -> dict:
    """
    Classify all (or only failed) comments for a video.

    Args:
        retry_failed:     When True, only re-process comments with classification_status="failed".
        auto_retry_count: How many automatic retries have already run. Managed internally —
                          do not set manually via the API.
    """
    logger.info(
        "classification_task_started",
        video_id=video_id,
        task_id=self.request.id,
        retry_failed=retry_failed,
        auto_retry_count=auto_retry_count,
    )

    try:
        result = asyncio.run(_run_classification(video_id, retry_failed=retry_failed))
        logger.info(
            "classification_task_completed",
            video_id=video_id,
            **{k: v for k, v in result.items() if k != "video_id"},
        )

        # ── Auto-retry: self-chain if failures remain ─────────────────────
        failed = result.get("failed", 0)
        if failed > 0 and auto_retry_count < MAX_AUTO_RETRIES:
            logger.info(
                "classification_auto_retry_dispatched",
                video_id=video_id,
                failed_remaining=failed,
                auto_retry_count=auto_retry_count + 1,
                max_auto_retries=MAX_AUTO_RETRIES,
            )
            classify_comments.apply_async(
                kwargs={
                    "video_id":         video_id,
                    "retry_failed":     True,
                    "auto_retry_count": auto_retry_count + 1,
                },
                queue     = "scraper",
                countdown = 3,   # small delay so MongoDB writes settle
            )
        elif failed > 0:
            logger.warning(
                "classification_auto_retry_exhausted",
                video_id=video_id,
                failed_remaining=failed,
                auto_retry_count=auto_retry_count,
            )

        return result

    except ValueError as exc:
        logger.error("classification_task_permanent_failure", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}

    except Exception as exc:
        if self.request.retries < self.max_retries:
            backoff = 60
            logger.warning(
                "classification_task_retrying",
                video_id=video_id,
                attempt=self.request.retries + 1,
                backoff=backoff,
                error=str(exc),
            )
            raise self.retry(
                exc=exc,
                countdown=backoff,
                kwargs={
                    "video_id":         video_id,
                    "retry_failed":     retry_failed,
                    "auto_retry_count": auto_retry_count,
                },
            )

        logger.error("classification_task_failed", video_id=video_id, error=str(exc))
        asyncio.run(_mark_failed(video_id, str(exc)))
        return {"video_id": video_id, "status": "failed", "error": str(exc)}


# ── Core async logic ──────────────────────────────────────────────────────────

async def _run_classification(video_id: str, retry_failed: bool = False) -> dict:
    mongo_client, db = make_db_client()
    try:
        comment_repo        = CommentRepository(db)
        classification_repo = ClassificationRepository(db)

        # ── Guard: summary must be completed ──────────────────────────────
        summary = await db["summaries"].find_one(
            {"video_id": video_id, "status": "completed"},
            {"_id": 0},
        )
        if not summary:
            raise ValueError(
                f"No completed summary for video {video_id!r}. "
                "Run POST /api/v1/summaries/{video_id} first."
            )

        # ── Load comments ──────────────────────────────────────────────────
        if retry_failed:
            # Load failed + all TLCs (need TLCs for parent-text map)
            failed_comments = await comment_repo.get_failed_for_classification(video_id)
            if not failed_comments:
                return {"video_id": video_id, "status": "completed", "total": 0,
                        "classified": 0, "skipped": 0, "failed": 0}
            all_tlcs = await comment_repo.get_all_for_classification(video_id)
            tlc_text_map: dict[str, str] = {
                c["comment_id"]: c.get("text", "")
                for c in all_tlcs
                if not c.get("is_reply", False)
            }
            target_comments = failed_comments
            logger.info("classification_retry_loading", video_id=video_id, failed=len(failed_comments))
        else:
            all_comments = await comment_repo.get_all_for_classification(video_id)
            total = len(all_comments)
            if total == 0:
                raise ValueError(
                    f"No comments found for video {video_id!r}. "
                    "Run the scrape job first."
                )
            tlc_text_map = {
                c["comment_id"]: c.get("text", "")
                for c in all_comments
                if not c.get("is_reply", False)
            }
            target_comments = all_comments
            logger.info("classification_loading_comments", video_id=video_id, total=total)

        # ── Attach parent_text to replies ──────────────────────────────────
        classification_input: list[dict] = []
        for c in target_comments:
            item: dict = {
                "comment_id": c["comment_id"],
                "text":       c.get("text", ""),
                "is_reply":   c.get("is_reply", False),
            }
            if c.get("is_reply") and c.get("parent_comment_id"):
                parent_text = tlc_text_map.get(c["parent_comment_id"], "")
                if parent_text:
                    item["parent_text"] = parent_text[:200]
            classification_input.append(item)

        # ── Mark as processing ─────────────────────────────────────────────
        await classification_repo.mark_processing(video_id, len(target_comments))

        # ── Run classification ─────────────────────────────────────────────
        settings   = get_settings()
        classifier = CommentClassifier(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
        results, skipped_count, failed_count = await classifier.classify_all(
            classification_input, summary
        )

        # ── Bulk-update comment docs ───────────────────────────────────────
        await comment_repo.bulk_update_classifications(video_id, results)

        # ── Compute and store aggregates ───────────────────────────────────
        if retry_failed:
            # Recompute from full DB state (includes previously classified comments)
            aggregates = await _recompute_aggregates_from_db(
                video_id, comment_repo, skipped_count, failed_count
            )
        else:
            aggregates = _compute_aggregates(results, len(target_comments), skipped_count, failed_count)
        await classification_repo.mark_completed(video_id, aggregates)

        return {
            "video_id":   video_id,
            "status":     "completed",
            "total":      len(target_comments),
            "classified": aggregates["classified_count"],
            "skipped":    skipped_count,
            "failed":     failed_count,
        }

    finally:
        mongo_client.close()


async def _recompute_aggregates_from_db(
    video_id: str,
    comment_repo: CommentRepository,
    skipped_count: int,
    failed_count: int,
) -> dict:
    """Recompute aggregates by reading all classified comments from DB."""
    counts = await comment_repo.get_classification_counts(video_id)
    classified_count = counts.get("total_done", 0)
    base = classified_count or 1

    sentiments = counts.get("sentiments", [])
    sentiment_counts: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for s in sentiments:
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    intent_keys = ["question", "praise", "criticism", "confusion",
                   "misconception", "request", "spam", "off_topic"]
    intent_counts: dict[str, int] = {k: 0 for k in intent_keys}
    for labels in counts.get("intent_labels", []):
        for label in (labels or []):
            if label in intent_counts:
                intent_counts[label] += 1

    def with_pct(c: dict, denom: int) -> dict:
        return {k: {"count": v, "pct": round(v / denom * 100, 1)} for k, v in c.items()}

    return {
        "classified_count":       classified_count,
        "failed_count":           failed_count,
        "skipped_count":          skipped_count,
        "sentiment_breakdown":    with_pct(sentiment_counts, base),
        "intent_breakdown":       with_pct(intent_counts, base),
        "computed_at":            datetime.now(timezone.utc),
        "classification_version": "v1",
    }


async def _mark_failed(video_id: str, error: str) -> None:
    mongo_client, db = make_db_client()
    try:
        await ClassificationRepository(db).mark_failed(video_id, error)
    finally:
        mongo_client.close()


# ── Aggregate computation ─────────────────────────────────────────────────────

def _compute_aggregates(
    results: list[dict],
    total_comments: int,
    skipped_count: int,
    failed_count: int,
) -> dict:
    """
    Compute sentiment + intent breakdowns from in-memory classification results.
    Runs in the Celery task — no extra DB round-trip needed.
    """
    classified = [r for r in results if r.get("classification_status") == "done"]
    classified_count = len(classified)
    base = classified_count or 1  # avoid div-by-zero

    # Sentiment counts
    sentiment_counts: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for r in classified:
        s = r.get("sentiment", "neutral")
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    # Intent counts (multi-label — one comment can increment multiple buckets)
    intent_keys = ["question", "praise", "criticism", "confusion",
                   "misconception", "request", "spam", "off_topic"]
    intent_counts: dict[str, int] = {k: 0 for k in intent_keys}
    for r in classified:
        for label in r.get("intent_labels", []):
            if label in intent_counts:
                intent_counts[label] += 1

    def with_pct(counts: dict, denominator: int) -> dict:
        return {
            k: {"count": v, "pct": round(v / denominator * 100, 1)}
            for k, v in counts.items()
        }

    return {
        "classified_count":    classified_count,
        "failed_count":        failed_count,
        "skipped_count":       skipped_count,
        "sentiment_breakdown": with_pct(sentiment_counts, base),
        "intent_breakdown":    with_pct(intent_counts, base),
        "computed_at":         datetime.now(timezone.utc),
        "classification_version": "v1",
    }
