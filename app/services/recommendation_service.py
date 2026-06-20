"""
app/services/recommendation_service.py
========================================
Phase 3D: Audience intelligence recommendation engine.

INPUT:  clusters (3C) + classified comments (3B) + video summary (3A)
OUTPUT: 4 types of rich, actionable recommendations

PIPELINE:
  1. Content gaps     — ranked by priority_score = comments × question_pct × (1 - gap_sim)
  2. Misconception map — group misconception comments by cluster, cross-ref key_claims
  3. Controversy hotspots — match criticism clusters against summary.controversy_triggers
  4. Unanswered questions — top by like_count, filtered to real topic clusters only
  5. Groq enrichment — ONE call with chain-of-thought + few-shot → rich recommendation text

PROMPT TECHNIQUES:
  - Expert persona: senior audience intelligence analyst
  - Chain-of-thought: model reasons before writing
  - Few-shot examples: 3 complete gap/misconception/controversy examples with BAD vs GOOD
  - Specificity forcing: required fields template
  - Anti-pattern examples: explicit BAD examples so model avoids vagueness
  - Context injection: video overview + key_claims + controversy_triggers
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from app.core.logging import get_logger
from app.services.relevance_filter import _get_model

logger = get_logger(__name__)

GROQ_BASE_URL           = "https://api.groq.com/openai/v1"
CONTROVERSY_MATCH_THRESHOLD  = 0.38   # cosine sim: cluster label vs controversy trigger
MISCONCEPTION_CLAIM_THRESHOLD = 0.60  # cosine sim: misconception text vs key_claim — high threshold to avoid wrong matches
MIN_MISCONCEPTION_LEN   = 30          # filter out troll/bot one-liners
MIN_UNANSWERED_LIKES    = 0           # include all, sort by likes
MAX_GAPS                = 5
MAX_MISCONCEPTIONS      = 5
MAX_CONTROVERSIES       = 4
MAX_UNANSWERED          = 8
MAX_PRAISE_PCT          = 50.0   # clusters where praise dominates are fan clusters, not topic gaps
HIGH_DEMAND_Q_PCT       = 40.0   # question_pct threshold to surface as gap even if gap_sim > 0.35
HIGH_DEMAND_GAP_SIM     = 0.42   # tighter than is_content_gap (0.35) — gap must be meaningfully uncovered
HIGH_DEMAND_MIN_COUNT   = 20     # minimum comment count to qualify as high-demand gap


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ContentGap:
    cluster_id:      int
    label:           str
    priority_score:  float
    comment_count:   int
    question_pct:    float
    gap_sim:         float
    keywords:        list[str]
    top_comments:    list[dict]
    what_to_do:      str = ""
    why:             str = ""
    suggested_hook:  str = ""
    urgency:         str = ""
    impact_type:     str = "new_video"


@dataclass
class MisconceptionItem:
    cluster_id:       int
    cluster_label:    str
    misconception_count: int
    related_claim:    str
    top_comments:     list[dict]
    what_to_do:       str = ""
    why:              str = ""
    suggested_hook:   str = ""
    urgency:          str = ""
    impact_type:      str = "pin_comment"


@dataclass
class ControversyHotspot:
    cluster_id:        int
    cluster_label:     str
    criticism_count:   int
    criticism_pct:     float
    matched_trigger:   str
    sentiment:         str
    top_comments:      list[dict]
    what_to_do:        str = ""
    why:               str = ""
    suggested_hook:    str = ""
    urgency:           str = ""
    impact_type:       str = "update_description"


@dataclass
class UnansweredQuestion:
    comment_id:   str
    text:         str
    like_count:   int
    cluster_id:   int
    cluster_label: str


@dataclass
class RecommendationResult:
    content_gaps:          list[ContentGap]
    misconceptions:        list[MisconceptionItem]
    controversy_hotspots:  list[ControversyHotspot]
    unanswered_questions:  list[UnansweredQuestion]


# ── Public API ────────────────────────────────────────────────────────────────

class RecommendationService:

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        self._api_key = api_key
        self._model   = model

    async def generate(
        self,
        clusters:               list[dict],
        summary:                dict,
        misconception_comments: list[dict],
        unanswered_comments:    list[dict],
    ) -> RecommendationResult:

        embed_model = _get_model()
        cluster_map = {cl["cluster_id"]: cl for cl in clusters}

        # ── Step 1: Content gaps ──────────────────────────────────────────
        gaps = _compute_content_gaps(clusters)
        logger.info("recommendations_gaps_computed", count=len(gaps))

        # ── Step 2: Misconception map ─────────────────────────────────────
        misconceptions = _compute_misconception_map(
            misconception_comments, cluster_map, summary, embed_model
        )
        logger.info("recommendations_misconceptions_computed", count=len(misconceptions))

        # ── Step 3: Controversy hotspots ──────────────────────────────────
        controversies = _compute_controversy_hotspots(clusters, summary, embed_model)
        logger.info("recommendations_controversies_computed", count=len(controversies))

        # ── Step 4: Top unanswered questions ─────────────────────────────
        unanswered = _compute_top_unanswered(unanswered_comments, cluster_map)
        logger.info("recommendations_unanswered_computed", count=len(unanswered))

        # ── Step 5: Groq enrichment ───────────────────────────────────────
        await _enrich_with_groq(
            gaps, misconceptions, controversies, summary, self._api_key, self._model
        )

        return RecommendationResult(
            content_gaps         = gaps,
            misconceptions       = misconceptions,
            controversy_hotspots = controversies,
            unanswered_questions = unanswered,
        )


# ── Step 1: Content gaps ──────────────────────────────────────────────────────

def _is_fan_cluster(cl: dict) -> bool:
    """True if the cluster is dominated by fan praise, not topic discussion."""
    intent_bd  = cl.get("intent_breakdown", {})
    praise_pct = intent_bd.get("praise", {}).get("pct", 0)
    return praise_pct >= MAX_PRAISE_PCT


def _compute_content_gaps(clusters: list[dict]) -> list[ContentGap]:
    seen_ids = set()
    gaps = []

    for cl in clusters:
        if cl.get("cluster_type") != "topic":
            continue
        if _is_fan_cluster(cl):
            continue

        intent_bd     = cl.get("intent_breakdown", {})
        question_data = intent_bd.get("question", {})
        question_pct  = question_data.get("pct", 0)
        comment_count = cl.get("comment_count", 0)
        gap_sim       = cl.get("gap_similarity_score", 0.0)
        cid           = cl["cluster_id"]

        is_flagged_gap   = cl.get("is_content_gap", False)
        is_high_demand   = (
            question_pct >= HIGH_DEMAND_Q_PCT
            and comment_count >= HIGH_DEMAND_MIN_COUNT
            and gap_sim < HIGH_DEMAND_GAP_SIM
        )

        if not is_flagged_gap and not is_high_demand:
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        priority_score = round(comment_count * (question_pct / 100) * (1 - gap_sim), 2)

        gaps.append(ContentGap(
            cluster_id     = cid,
            label          = cl.get("label", ""),
            priority_score = priority_score,
            comment_count  = comment_count,
            question_pct   = round(question_pct, 1),
            gap_sim        = gap_sim,
            keywords       = cl.get("keywords", [])[:6],
            top_comments   = cl.get("top_comments", [])[:3],
        ))

    gaps.sort(key=lambda g: g.priority_score, reverse=True)
    return gaps[:MAX_GAPS]


# ── Step 2: Misconception map ─────────────────────────────────────────────────

def _compute_misconception_map(
    misconception_comments: list[dict],
    cluster_map:            dict,
    summary:                dict,
    embed_model,
) -> list[MisconceptionItem]:

    key_claims = summary.get("key_claims", [])
    if not key_claims:
        return []

    # Embed key claims once
    claim_embeddings = embed_model.encode(
        key_claims, show_progress_bar=False, convert_to_numpy=True
    )

    # Group misconception comments by cluster, filter noise
    cluster_groups: dict[int, list[dict]] = {}
    for c in misconception_comments:
        cid = c.get("cluster_id", -1)
        if cid == -1:
            continue
        cl = cluster_map.get(cid)
        if not cl or cl.get("cluster_type") != "topic":
            continue
        if _is_fan_cluster(cl):
            continue
        text = (c.get("text") or "").strip()
        if len(text) < MIN_MISCONCEPTION_LEN:
            continue
        cluster_groups.setdefault(cid, []).append(c)

    results = []
    for cluster_id, comments in cluster_groups.items():
        cl = cluster_map[cluster_id]

        # Find which key_claim this misconception cluster is about
        sample_texts  = [c.get("text", "")[:200] for c in comments[:5]]
        combined_text = " ".join(sample_texts)
        misc_emb      = embed_model.encode([combined_text], convert_to_numpy=True)
        sims          = cosine_similarity(misc_emb, claim_embeddings)[0]
        best_idx      = int(np.argmax(sims))
        best_sim      = float(sims[best_idx])
        related_claim = key_claims[best_idx] if best_sim >= MISCONCEPTION_CLAIM_THRESHOLD else ""

        top3 = sorted(comments, key=lambda x: x.get("like_count") or 0, reverse=True)[:3]

        results.append(MisconceptionItem(
            cluster_id          = cluster_id,
            cluster_label       = cl.get("label", ""),
            misconception_count = len(comments),
            related_claim       = related_claim,
            top_comments        = [_slim(c) for c in top3],
        ))

    results.sort(key=lambda m: m.misconception_count, reverse=True)
    return results[:MAX_MISCONCEPTIONS]


# ── Step 3: Controversy hotspots ──────────────────────────────────────────────

def _compute_controversy_hotspots(
    clusters:   list[dict],
    summary:    dict,
    embed_model,
) -> list[ControversyHotspot]:

    triggers = summary.get("controversy_triggers", [])
    if not triggers:
        return []

    trigger_embeddings = embed_model.encode(
        triggers, show_progress_bar=False, convert_to_numpy=True
    )

    results = []
    for cl in clusters:
        if cl.get("cluster_type") != "topic":
            continue

        intent_bd      = cl.get("intent_breakdown", {})
        criticism_data = intent_bd.get("criticism", {})
        criticism_pct  = criticism_data.get("pct", 0)
        criticism_count = criticism_data.get("count", 0)

        if criticism_pct < 25:   # must be meaningfully criticism-heavy
            continue

        # Match against controversy triggers
        label_emb = embed_model.encode([cl.get("label", "")], convert_to_numpy=True)
        sims      = cosine_similarity(label_emb, trigger_embeddings)[0]
        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])

        matched_trigger = triggers[best_idx] if best_sim >= CONTROVERSY_MATCH_THRESHOLD else \
                          f"Unmatched controversy in '{cl.get('label', '')}'"

        sentiment_bd = cl.get("sentiment_breakdown", {})
        dominant_sent = max(sentiment_bd.items(), key=lambda x: x[1]["count"])[0] \
                        if sentiment_bd else "neutral"

        # A cluster with positive dominant sentiment is not a controversy —
        # criticism_pct can be inflated by a few vocal critics in an otherwise
        # supportive cluster. Only flag when the audience is genuinely negative/divided.
        if dominant_sent == "positive":
            continue

        results.append(ControversyHotspot(
            cluster_id      = cl["cluster_id"],
            cluster_label   = cl.get("label", ""),
            criticism_count = criticism_count,
            criticism_pct   = criticism_pct,
            matched_trigger = matched_trigger,
            sentiment       = dominant_sent,
            top_comments    = cl.get("top_comments", [])[:3],
        ))

    results.sort(key=lambda c: c.criticism_count, reverse=True)
    return results[:MAX_CONTROVERSIES]


# ── Step 4: Top unanswered questions ─────────────────────────────────────────

def _compute_top_unanswered(
    comments:    list[dict],
    cluster_map: dict,
) -> list[UnansweredQuestion]:
    results = []
    for c in comments:
        cid = c.get("cluster_id", -1)
        cl  = cluster_map.get(cid)
        if not cl or cl.get("cluster_type") != "topic":
            continue
        if _is_fan_cluster(cl):
            continue
        text = (c.get("text") or "").strip()
        if len(text) < 15:
            continue
        results.append(UnansweredQuestion(
            comment_id    = c["comment_id"],
            text          = text[:300],
            like_count    = c.get("like_count") or 0,
            cluster_id    = cid,
            cluster_label = cl.get("label", ""),
        ))

    results.sort(key=lambda q: q.like_count, reverse=True)
    return results[:MAX_UNANSWERED]


# ── Step 5: Groq enrichment ───────────────────────────────────────────────────

async def _enrich_with_groq(
    gaps:          list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies: list[ControversyHotspot],
    summary:       dict,
    api_key:       str,
    model:         str,
) -> None:
    """
    Single Groq call with chain-of-thought + few-shot.
    Enriches gaps, misconceptions, controversies in-place with:
    what_to_do, why, suggested_hook, urgency, impact_type.
    """
    if not gaps and not misconceptions and not controversies:
        return

    client  = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    overview = summary.get("overview", "")
    claims   = summary.get("key_claims", [])[:8]

    system_prompt = _build_system_prompt()
    user_prompt   = _build_user_prompt(gaps, misconceptions, controversies, overview, claims)

    try:
        response = await client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format = {"type": "json_object"},
            max_tokens  = 2000,
            temperature = 0.3,
        )
        data = json.loads(response.choices[0].message.content)
        logger.info("recommendations_groq_done")
        _apply_enrichment(data, gaps, misconceptions, controversies)
    except Exception as exc:
        logger.warning("recommendations_groq_failed", error=str(exc))
        _apply_fallback_labels(gaps, misconceptions, controversies)


def _build_system_prompt() -> str:
    return """You are a senior YouTube audience intelligence analyst with 10+ years helping \
educational content creators grow. You convert raw comment data into specific, high-impact \
content decisions that creators can act on TODAY.

QUALITY BAR — every recommendation must pass this test:
"Can the creator read this and know EXACTLY what to do, why it matters, and how to do it?"

❌ BAD (too vague, never write this):
"Consider addressing supplement questions as there seems to be audience interest."

✅ GOOD (specific + evidence + action):
"130 viewers are discussing NAC eye drops with 78 asking the same question: what dose and brand. \
Create a 10-min video titled 'The Exact NAC Protocol I Recommend (Brand + Dose + Timing)'. \
This is your clearest content gap with zero competition coverage."

REQUIRED FIELDS per recommendation:
- what_to_do: One clear sentence — the specific action
- why: Evidence-backed reasoning citing comment counts or verbatim examples
- suggested_hook: A specific video title OR pinned comment text they can copy-paste
- urgency: "high" (100+ comments or strong negative), "medium" (50-99), "low" (<50)
- impact_type: "new_video" | "pin_comment" | "update_description" | "community_post"

FEW-SHOT EXAMPLES:

--- CONTENT GAP EXAMPLE ---
Finding: 50 comments all about astigmatism, 100% questions, not in video
{
  "what_to_do": "Create a dedicated video on astigmatism — natural causes and whether any of your 13 remedies apply",
  "why": "50 viewers asked about astigmatism in your comments. Every single one is a question with no answer from the video. Sample: 'Does any of this help with astigmatism? My optometrist says nothing helps.' This is pure unmet demand.",
  "suggested_hook": "Can You Actually Fix Astigmatism Naturally? (What Eye Doctors Won't Tell You)",
  "urgency": "medium",
  "impact_type": "new_video"
}

--- MISCONCEPTION EXAMPLE ---
Finding: 18 comments claim NAC permanently cures cataracts; video said it slows progression
{
  "what_to_do": "Pin a comment correcting the NAC permanent cure misconception before it spreads further",
  "why": "18 viewers believe NAC completely reverses cataracts ('NAC cured my cataracts completely'). Your video stated NAC helps slow glycation — not permanent reversal. This misinformation risks your medical credibility and could cause viewers to delay real treatment.",
  "suggested_hook": "CLARIFICATION: NAC eye drops slow cataract progression — they don't permanently reverse advanced cataracts. Always consult an ophthalmologist for advanced cases.",
  "urgency": "high",
  "impact_type": "pin_comment"
}

--- CONTROVERSY EXAMPLE ---
Finding: 67 comments criticizing LED light safety claims, negative sentiment dominant
{
  "what_to_do": "Add peer-reviewed sources to the video description for your LED light and eye health claims",
  "why": "67 viewers are actively challenging your LED claims in comments ('There's zero peer-reviewed evidence for this'). Skeptics are discrediting the video in replies. Adding 2-3 sources converts skeptics to believers and protects your credibility.",
  "suggested_hook": "Sources: [LED flicker and eye strain - Journal of Optometry 2022] [Blue light effects on melatonin - Sleep Medicine Reviews]",
  "urgency": "high",
  "impact_type": "update_description"
}"""


def _build_user_prompt(
    gaps:          list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies: list[ControversyHotspot],
    overview:      str,
    claims:        list[str],
) -> str:
    lines = [
        f"VIDEO OVERVIEW: {overview}",
        f"KEY CLAIMS IN VIDEO: {'; '.join(claims[:6])}",
        "",
        "THINK STEP BY STEP before writing each recommendation:",
        "1. What does this data tell us about the audience's knowledge gap or pain point?",
        "2. What is the creator missing or getting wrong?",
        "3. What single action would have the highest impact for the creator?",
        "Then write the recommendation.",
        "",
    ]

    if gaps:
        lines.append("=== CONTENT GAPS (topics audience cares about that video didn't cover) ===")
        for g in gaps:
            sample = g.top_comments[0]["text"][:120] if g.top_comments else ""
            lines.append(
                f'Gap "{g.label}" (id:{g.cluster_id}): {g.comment_count} comments, '
                f'{g.question_pct}% questions, gap_score={g.gap_sim}, '
                f'keywords=[{", ".join(g.keywords[:5])}], '
                f'sample="{sample}"'
            )
        lines.append("")

    if misconceptions:
        lines.append("=== MISCONCEPTIONS (wrong beliefs spreading in comments) ===")
        for m in misconceptions:
            top_text = m.top_comments[0]["text"][:120] if m.top_comments else ""
            claim_str = f'related_claim="{m.related_claim}"' if m.related_claim else "claim=unmatched"
            lines.append(
                f'Misconception in "{m.cluster_label}" (id:{m.cluster_id}): '
                f'{m.misconception_count} misconception comments, {claim_str}, '
                f'example="{top_text}"'
            )
        lines.append("")

    if controversies:
        lines.append("=== CONTROVERSY HOTSPOTS (audience pushing back on claims) ===")
        for c in controversies:
            top_text = c.top_comments[0]["text"][:120] if c.top_comments else ""
            lines.append(
                f'Controversy in "{c.cluster_label}" (id:{c.cluster_id}): '
                f'{c.criticism_count} critical comments ({c.criticism_pct}% of cluster), '
                f'trigger="{c.matched_trigger}", sentiment={c.sentiment}, '
                f'example="{top_text}"'
            )
        lines.append("")

    lines.append(
        'Return JSON with this exact structure:\n'
        '{\n'
        '  "content_gaps": [{"cluster_id": <int>, "what_to_do": "", "why": "", "suggested_hook": "", "urgency": "", "impact_type": ""}],\n'
        '  "misconceptions": [{"cluster_id": <int>, "what_to_do": "", "why": "", "suggested_hook": "", "urgency": "", "impact_type": ""}],\n'
        '  "controversy_hotspots": [{"cluster_id": <int>, "what_to_do": "", "why": "", "suggested_hook": "", "urgency": "", "impact_type": ""}]\n'
        '}'
    )

    return "\n".join(lines)


def _apply_enrichment(
    data:          dict,
    gaps:          list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies: list[ControversyHotspot],
) -> None:
    def _apply(items, key):
        enriched = {str(item["cluster_id"]): item for item in data.get(key, [])}
        for obj in items:
            e = enriched.get(str(obj.cluster_id), {})
            obj.what_to_do     = e.get("what_to_do", "")
            obj.why            = e.get("why", "")
            obj.suggested_hook = e.get("suggested_hook", "")
            obj.urgency        = e.get("urgency", "medium")
            obj.impact_type    = e.get("impact_type", "new_video")

    _apply(gaps,           "content_gaps")
    _apply(misconceptions, "misconceptions")
    _apply(controversies,  "controversy_hotspots")


def _apply_fallback_labels(gaps, misconceptions, controversies) -> None:
    for g in gaps:
        g.what_to_do = f"Create content addressing '{g.label}'"
        g.urgency    = "high" if g.comment_count >= 100 else "medium"
    for m in misconceptions:
        m.what_to_do = f"Address misconceptions in '{m.cluster_label}'"
        m.urgency    = "high" if m.misconception_count >= 20 else "medium"
    for c in controversies:
        c.what_to_do = f"Add sources for claims in '{c.cluster_label}'"
        c.urgency    = "high" if c.criticism_count >= 50 else "medium"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slim(c: dict) -> dict:
    return {
        "comment_id": c.get("comment_id", ""),
        "text":       (c.get("text") or "")[:200],
        "like_count": c.get("like_count") or 0,
    }
