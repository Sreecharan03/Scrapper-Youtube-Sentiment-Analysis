"""
app/services/dashboard_service.py
===================================
Phase 3F: Dashboard aggregation — no LLM calls, pure data joins.

Reads 6 collections in parallel, computes three scores, shapes into
DashboardResponse. One DB round-trip per collection (asyncio.gather).

SCORE FORMULAS:
  health_score:
    positive_sentiment_pct  × 0.50   (audience feels good)
    (100 - misconception_pct) × 0.30 (low misinformation pressure)
    (100 - criticism_pct)    × 0.20  (low hostility)
    → 0-100, rounded to int

  opportunity_score:
    = top video idea's demand_score (0-100)
    Fallback if no ideas: top gap priority_score normalised by cap (200)
    Demand_score is already calibrated in recommendation_service._extract_strategic()

  risk_score:
    criticism_pct  × 0.60
    misconception_pct × 0.40
    → 0-100, rounded to int
"""

import asyncio

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger

logger = get_logger(__name__)

OPPORTUNITY_PRIORITY_CAP = 200.0   # max realistic priority_score for normalisation

# Intent order for tab display (most actionable first)
INTENT_TAB_ORDER = ["question", "criticism", "misconception", "confusion", "request", "praise"]


async def build_dashboard(video_id: str, db: AsyncIOMotorDatabase) -> dict:
    """
    Aggregate all collections for a single video into a dashboard dict.
    Returns a plain dict — endpoint converts to DashboardResponse.
    Raises ValueError if minimum required data (classification) is missing.
    """
    (
        video_doc,
        job_doc,
        analysis_doc,
        rec_doc,
        intent_summary_doc,
        cluster_info_doc,
    ) = await asyncio.gather(
        db["videos"].find_one(         {"video_id": video_id}, {"_id": 0}),
        db["jobs"].find_one(           {"video_id": video_id}, {"_id": 0, "video_url": 1, "comments_collected": 1, "status": 1}),
        db["comment_analysis"].find_one({"video_id": video_id}, {"_id": 0}),
        db["recommendations"].find_one( {"video_id": video_id}, {"_id": 0}),
        db["intent_summaries"].find_one({"video_id": video_id}, {"_id": 0}),
        db["cluster_info"].find_one(    {"video_id": video_id}, {"_id": 0, "status": 1}),
    )

    if not analysis_doc or analysis_doc.get("status") != "completed":
        raise ValueError(
            f"Classification not completed for video_id={video_id}. "
            "Run POST /api/v1/analysis/{video_id} first."
        )

    # ── Video header ──────────────────────────────────────────────────────────
    comments_collected = (job_doc or {}).get("comments_collected", 0)
    url = (job_doc or {}).get("video_url") or f"https://www.youtube.com/watch?v={video_id}"

    video = {
        "video_id":           video_id,
        "url":                url,
        "title":              (video_doc or {}).get("title"),
        "channel_name":       (video_doc or {}).get("channel_name"),
        "view_count":         (video_doc or {}).get("view_count"),
        "like_count":         (video_doc or {}).get("like_count"),
        "comments_collected": comments_collected,
    }

    # ── Intent + sentiment breakdowns ─────────────────────────────────────────
    raw_intent    = analysis_doc.get("intent_breakdown", {})
    raw_sentiment = analysis_doc.get("sentiment_breakdown", {})
    classified    = analysis_doc.get("classified_count", 0)

    intent_bd   = _parse_breakdown(raw_intent)
    sentiment_bd = _parse_breakdown(raw_sentiment)

    # Pull key percentages for score computation
    positive_pct     = raw_sentiment.get("positive", {}).get("pct", 0.0)
    criticism_pct    = raw_intent.get("criticism", {}).get("pct", 0.0)
    misconception_pct = raw_intent.get("misconception", {}).get("pct", 0.0)

    # ── Scores ────────────────────────────────────────────────────────────────
    health_score = _health(positive_pct, criticism_pct, misconception_pct)
    risk_score   = _risk(criticism_pct, misconception_pct)

    # Opportunity: prefer top video idea demand_score, fallback to gap
    top_ideas   = (rec_doc or {}).get("top_video_ideas", [])
    top_gaps    = (rec_doc or {}).get("content_gaps", [])
    opportunity_score = _opportunity(top_ideas, top_gaps)

    # ── Recommendations layer ─────────────────────────────────────────────────
    executive_summary       = (rec_doc or {}).get("executive_summary", "")
    audience_stage          = (rec_doc or {}).get("audience_stage", "")
    audience_mood           = (rec_doc or {}).get("audience_mood", "")
    purchase_intent_signals = (rec_doc or {}).get("purchase_intent_signals", [])
    content_series          = (rec_doc or {}).get("content_series", [])
    risk_alerts             = (rec_doc or {}).get("risk_alerts", [])
    generated_at            = (rec_doc or {}).get("generated_at")

    # Top 3 ideas for hero panel (frontend shows all 5 in deep dive)
    top_3_video_ideas = [
        _shape_idea(v) for v in top_ideas[:3]
    ]

    # ── Intent tabs ───────────────────────────────────────────────────────────
    raw_intent_summaries = (intent_summary_doc or {}).get("intent_summaries", {})
    overall_summary      = (intent_summary_doc or {}).get("overall_summary", "")

    intent_tabs = []
    for intent in INTENT_TAB_ORDER:
        entry = raw_intent_summaries.get(intent, {})
        count = entry.get("count", 0) or raw_intent.get(intent, {}).get("count", 0)
        if count == 0:
            continue
        intent_tabs.append({
            "intent":  intent,
            "count":   count,
            "summary": entry.get("summary", ""),
        })

    # ── Deep dive items ───────────────────────────────────────────────────────
    content_gaps        = [_shape_gap(g)  for g in (rec_doc or {}).get("content_gaps", [])]
    misconceptions      = [_shape_misc(m) for m in (rec_doc or {}).get("misconceptions", [])]
    controversy_hotspots = [_shape_cont(c) for c in (rec_doc or {}).get("controversy_hotspots", [])]
    unanswered_questions = [_shape_unanswered(q) for q in (rec_doc or {}).get("unanswered_questions", [])]

    # ── Pipeline status ───────────────────────────────────────────────────────
    pipeline_status = {
        "classification":   analysis_doc.get("status", "missing"),
        "clustering":       (cluster_info_doc or {}).get("status", "missing"),
        "recommendations":  (rec_doc or {}).get("status", "missing"),
        "intent_summaries": (intent_summary_doc or {}).get("status", "missing"),
    }

    logger.info(
        "dashboard_built",
        video_id        = video_id,
        health          = health_score,
        opportunity     = opportunity_score,
        risk            = risk_score,
        intent_tabs     = len(intent_tabs),
        video_ideas     = len(top_ideas),
        gaps            = len(content_gaps),
        misconceptions  = len(misconceptions),
        controversies   = len(controversy_hotspots),
    )

    return {
        "video":                 video,
        "scores": {
            "health":      health_score,
            "opportunity": opportunity_score,
            "risk":        risk_score,
        },
        "executive_summary":       executive_summary,
        "audience_stage":          audience_stage,
        "audience_mood":           audience_mood,
        "overall_summary":         overall_summary,
        "top_video_ideas":         top_3_video_ideas,
        "intent_breakdown":        {**intent_bd, "total": classified},
        "sentiment_breakdown":     sentiment_bd,
        "intent_tabs":             intent_tabs,
        "content_gaps":            content_gaps,
        "misconceptions":          misconceptions,
        "controversy_hotspots":    controversy_hotspots,
        "unanswered_questions":    unanswered_questions,
        "purchase_intent_signals": purchase_intent_signals,
        "content_series":          content_series,
        "risk_alerts":             risk_alerts,
        "pipeline_status":         pipeline_status,
        "generated_at":            generated_at,
    }


# ── Score helpers ─────────────────────────────────────────────────────────────

def _health(positive_pct: float, criticism_pct: float, misconception_pct: float) -> int:
    score = (
        positive_pct     * 0.50
        + (100 - misconception_pct) * 0.30
        + (100 - criticism_pct)     * 0.20
    )
    return round(min(100, max(0, score)))


def _opportunity(top_ideas: list, top_gaps: list) -> int:
    if top_ideas:
        return int(top_ideas[0].get("demand_score", 0))
    if top_gaps:
        priority = top_gaps[0].get("priority_score", 0)
        return round(min(100, priority / OPPORTUNITY_PRIORITY_CAP * 100))
    return 0


def _risk(criticism_pct: float, misconception_pct: float) -> int:
    score = criticism_pct * 0.60 + misconception_pct * 0.40
    return round(min(100, max(0, score)))


# ── Shape helpers ─────────────────────────────────────────────────────────────

def _parse_breakdown(raw: dict) -> dict:
    return {
        k: {"count": v.get("count", 0), "pct": v.get("pct", 0.0)}
        for k, v in raw.items()
        if isinstance(v, dict)
    }


def _slim_comments(raw_list: list, limit: int = 2) -> list[dict]:
    return [
        {
            "comment_id": c.get("comment_id", ""),
            "text":       (c.get("text") or "")[:200],
            "like_count": c.get("like_count") or 0,
        }
        for c in (raw_list or [])[:limit]
    ]


def _shape_idea(v: dict) -> dict:
    return {
        "rank":           v.get("rank", 0),
        "title":          v.get("title", ""),
        "demand_score":   v.get("demand_score", 0),
        "confidence_pct": v.get("confidence_pct", 0),
        "why":            v.get("why", ""),
        "evidence_count": v.get("evidence_count", 0),
        "format":         v.get("format", "long_video"),
    }


def _shape_gap(g: dict) -> dict:
    return {
        "cluster_id":     g.get("cluster_id", 0),
        "label":          g.get("label", ""),
        "comment_count":  g.get("comment_count", 0),
        "question_pct":   g.get("question_pct", 0.0),
        "priority_score": g.get("priority_score", 0.0),
        "what_to_do":     g.get("what_to_do", ""),
        "suggested_hook": g.get("suggested_hook", ""),
        "urgency":        g.get("urgency", "medium"),
        "top_comments":   _slim_comments(g.get("top_comments", [])),
    }


def _shape_misc(m: dict) -> dict:
    return {
        "cluster_id":          m.get("cluster_id", 0),
        "cluster_label":       m.get("cluster_label", ""),
        "misconception_count": m.get("misconception_count", 0),
        "related_claim":       m.get("related_claim", ""),
        "what_to_do":          m.get("what_to_do", ""),
        "suggested_hook":      m.get("suggested_hook", ""),
        "urgency":             m.get("urgency", "medium"),
        "top_comments":        _slim_comments(m.get("top_comments", [])),
    }


def _shape_cont(c: dict) -> dict:
    return {
        "cluster_id":      c.get("cluster_id", 0),
        "cluster_label":   c.get("cluster_label", ""),
        "criticism_count": c.get("criticism_count", 0),
        "criticism_pct":   c.get("criticism_pct", 0.0),
        "matched_trigger": c.get("matched_trigger", ""),
        "what_to_do":      c.get("what_to_do", ""),
        "suggested_hook":  c.get("suggested_hook", ""),
        "urgency":         c.get("urgency", "medium"),
        "top_comments":    _slim_comments(c.get("top_comments", [])),
    }


def _shape_unanswered(q: dict) -> dict:
    return {
        "comment_id":    q.get("comment_id", ""),
        "text":          (q.get("text") or "")[:300],
        "like_count":    q.get("like_count") or 0,
        "cluster_label": q.get("cluster_label", ""),
    }
