"""
app/services/recommendation_service.py
========================================
Phase 3D: Audience intelligence recommendation engine.

PROMPT ARCHITECTURE (production-grade):

  Two sequential Groq calls, two different models:

  Call 1 — Strategic layer  (llama-3.3-70b-versatile)
    Techniques used:
    · Quoted data block  — all numbers injected verbatim so model cannot hallucinate them
    · Enumerated slots   — server pre-selects top 5 clusters; LLM converts, does not choose
    · CoT scratchpad     — model reasons in "reasoning" field before committing to answers
    · Verification field — model echoes back locked stats; mismatch = hallucination detected
    · No fabrication rule — hooks anchored to KEY CLAIMS only; no journal citations

  Call 2 — Per-item enrichment  (llama-3.1-8b-instant)
    Templated task: what_to_do / why / suggested_hook / urgency / impact_type per item
    Hooks must cite only KEY CLAIMS; scientific corrections forbidden without claim backing

PRINCIPLE:
  LLM writes narrative, server provides data.
  Server computes WHICH clusters become video ideas (sort by demand signal).
  LLM decides HOW to describe them (title, why, demand_score, format).
"""

import json
from dataclasses import dataclass, field

import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from app.core.logging import get_logger
from app.services.relevance_filter import _get_model

logger = get_logger(__name__)

GROQ_BASE_URL                = "https://api.groq.com/openai/v1"
CONTROVERSY_MATCH_THRESHOLD  = 0.38
MISCONCEPTION_CLAIM_THRESHOLD = 0.60
MIN_MISCONCEPTION_LEN        = 30
MAX_GAPS                     = 5
MAX_MISCONCEPTIONS           = 5
MAX_CONTROVERSIES            = 4
MAX_UNANSWERED               = 8
MAX_VIDEO_IDEAS              = 5
MAX_PRAISE_PCT               = 50.0
HIGH_DEMAND_Q_PCT            = 40.0
HIGH_DEMAND_GAP_SIM          = 0.42
HIGH_DEMAND_MIN_COUNT        = 20

VALID_STAGES  = {"Awareness", "Interest", "Implementation", "Evaluation", "Advocacy"}
VALID_MOODS   = {"Excited", "Curious", "Confused", "Skeptical", "Satisfied"}
VALID_FORMATS = {"long_video", "short", "live_qa", "community_post"}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ContentGap:
    cluster_id:     int
    label:          str
    priority_score: float
    comment_count:  int
    question_pct:   float
    gap_sim:        float
    keywords:       list[str]
    top_comments:   list[dict]
    what_to_do:     str = ""
    why:            str = ""
    suggested_hook: str = ""
    urgency:        str = ""
    impact_type:    str = "new_video"


@dataclass
class MisconceptionItem:
    cluster_id:          int
    cluster_label:       str
    misconception_count: int
    related_claim:       str
    top_comments:        list[dict]
    what_to_do:          str = ""
    why:                 str = ""
    suggested_hook:      str = ""
    urgency:             str = ""
    impact_type:         str = "pin_comment"


@dataclass
class ControversyHotspot:
    cluster_id:      int
    cluster_label:   str
    criticism_count: int
    criticism_pct:   float
    matched_trigger: str
    sentiment:       str
    top_comments:    list[dict]
    what_to_do:      str = ""
    why:             str = ""
    suggested_hook:  str = ""
    urgency:         str = ""
    impact_type:     str = "update_description"


@dataclass
class UnansweredQuestion:
    comment_id:    str
    text:          str
    like_count:    int
    cluster_id:    int
    cluster_label: str


@dataclass
class VideoIdea:
    rank:           int
    title:          str
    demand_score:   int
    confidence_pct: int
    why:            str
    evidence_count: int
    format:         str


@dataclass
class RecommendationResult:
    content_gaps:           list[ContentGap]
    misconceptions:         list[MisconceptionItem]
    controversy_hotspots:   list[ControversyHotspot]
    unanswered_questions:   list[UnansweredQuestion]
    executive_summary:       str             = ""
    audience_stage:          str             = ""
    audience_mood:           str             = ""
    top_video_ideas:         list[VideoIdea] = field(default_factory=list)
    purchase_intent_signals: list[str]       = field(default_factory=list)
    content_series:          list[str]       = field(default_factory=list)
    risk_alerts:             list[str]       = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

class RecommendationService:

    def __init__(
        self,
        api_key:          str,
        model:            str = "llama-3.1-8b-instant",
        strategic_model:  str = "llama-3.3-70b-versatile",
    ) -> None:
        self._api_key         = api_key
        self._model           = model            # 8b — per-item enrichment
        self._strategic_model = strategic_model  # 70b — strategic layer

    async def generate(
        self,
        clusters:               list[dict],
        summary:                dict,
        misconception_comments: list[dict],
        unanswered_comments:    list[dict],
        intent_counts:          dict,
    ) -> RecommendationResult:

        embed_model = _get_model()
        cluster_map = {cl["cluster_id"]: cl for cl in clusters}

        gaps           = _compute_content_gaps(clusters)
        misconceptions = _compute_misconception_map(
            misconception_comments, cluster_map, summary, embed_model
        )
        controversies  = _compute_controversy_hotspots(clusters, summary, embed_model)
        unanswered     = _compute_top_unanswered(unanswered_comments, cluster_map)

        logger.info(
            "recommendations_computed",
            gaps=len(gaps), misconceptions=len(misconceptions),
            controversies=len(controversies), unanswered=len(unanswered),
        )

        # Pre-select video slots server-side — LLM converts, does not choose
        video_slots = _select_video_slots(clusters, gaps)

        strategic = await _call_strategic_layer(
            video_slots, intent_counts, gaps, misconceptions, controversies,
            clusters, summary, self._api_key, self._strategic_model,
        )

        if gaps or misconceptions or controversies:
            await _call_per_item_enrichment(
                gaps, misconceptions, controversies, summary,
                self._api_key, self._model,
            )

        return RecommendationResult(
            content_gaps            = gaps,
            misconceptions          = misconceptions,
            controversy_hotspots    = controversies,
            unanswered_questions    = unanswered,
            executive_summary       = strategic.get("executive_summary", ""),
            audience_stage          = strategic.get("audience_stage", ""),
            audience_mood           = strategic.get("audience_mood", ""),
            top_video_ideas         = strategic.get("top_video_ideas", []),
            purchase_intent_signals = strategic.get("purchase_intent_signals", []),
            content_series          = strategic.get("content_series", []),
            risk_alerts             = strategic.get("risk_alerts", []),
        )


# ── Computation steps ─────────────────────────────────────────────────────────

def _is_fan_cluster(cl: dict) -> bool:
    return cl.get("intent_breakdown", {}).get("praise", {}).get("pct", 0) >= MAX_PRAISE_PCT


def _compute_content_gaps(clusters: list[dict]) -> list[ContentGap]:
    seen_ids, gaps = set(), []
    for cl in clusters:
        if cl.get("cluster_type") != "topic" or _is_fan_cluster(cl):
            continue
        bd           = cl.get("intent_breakdown", {})
        question_pct = bd.get("question", {}).get("pct", 0)
        count        = cl.get("comment_count", 0)
        gap_sim      = cl.get("gap_similarity_score", 0.0)
        cid          = cl["cluster_id"]

        if not cl.get("is_content_gap") and not (
            question_pct >= HIGH_DEMAND_Q_PCT and count >= HIGH_DEMAND_MIN_COUNT and gap_sim < HIGH_DEMAND_GAP_SIM
        ):
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        gaps.append(ContentGap(
            cluster_id     = cid,
            label          = cl.get("label", ""),
            priority_score = round(count * (question_pct / 100) * (1 - gap_sim), 2),
            comment_count  = count,
            question_pct   = round(question_pct, 1),
            gap_sim        = gap_sim,
            keywords       = cl.get("keywords", [])[:6],
            top_comments   = cl.get("top_comments", [])[:3],
        ))

    gaps.sort(key=lambda g: g.priority_score, reverse=True)
    return gaps[:MAX_GAPS]


def _compute_misconception_map(
    comments: list[dict], cluster_map: dict, summary: dict, embed_model
) -> list[MisconceptionItem]:
    key_claims = summary.get("key_claims", [])
    if not key_claims:
        return []

    claim_embs = embed_model.encode(key_claims, show_progress_bar=False, convert_to_numpy=True)
    groups: dict[int, list[dict]] = {}

    for c in comments:
        cid = c.get("cluster_id", -1)
        cl  = cluster_map.get(cid)
        if not cl or cl.get("cluster_type") != "topic" or _is_fan_cluster(cl):
            continue
        if len((c.get("text") or "").strip()) < MIN_MISCONCEPTION_LEN:
            continue
        groups.setdefault(cid, []).append(c)

    results = []
    for cid, coms in groups.items():
        cl       = cluster_map[cid]
        sample   = " ".join(c.get("text", "")[:200] for c in coms[:5])
        misc_emb = embed_model.encode([sample], convert_to_numpy=True)
        sims     = cosine_similarity(misc_emb, claim_embs)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        top3     = sorted(coms, key=lambda x: x.get("like_count") or 0, reverse=True)[:3]
        results.append(MisconceptionItem(
            cluster_id          = cid,
            cluster_label       = cl.get("label", ""),
            misconception_count = len(coms),
            related_claim       = key_claims[best_idx] if best_sim >= MISCONCEPTION_CLAIM_THRESHOLD else "",
            top_comments        = [_slim(c) for c in top3],
        ))

    results.sort(key=lambda m: m.misconception_count, reverse=True)
    return results[:MAX_MISCONCEPTIONS]


def _compute_controversy_hotspots(
    clusters: list[dict], summary: dict, embed_model
) -> list[ControversyHotspot]:
    triggers = summary.get("controversy_triggers", [])
    if not triggers:
        return []

    trig_embs = embed_model.encode(triggers, show_progress_bar=False, convert_to_numpy=True)
    results   = []

    for cl in clusters:
        if cl.get("cluster_type") != "topic":
            continue
        bd         = cl.get("intent_breakdown", {})
        crit_pct   = bd.get("criticism", {}).get("pct", 0)
        crit_count = bd.get("criticism", {}).get("count", 0)
        if crit_pct < 25:
            continue

        label_emb = embed_model.encode([cl.get("label", "")], convert_to_numpy=True)
        sims      = cosine_similarity(label_emb, trig_embs)[0]
        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])

        sent_bd  = cl.get("sentiment_breakdown", {})
        dominant = max(sent_bd.items(), key=lambda x: x[1]["count"])[0] if sent_bd else "neutral"
        if dominant == "positive":
            continue

        results.append(ControversyHotspot(
            cluster_id      = cl["cluster_id"],
            cluster_label   = cl.get("label", ""),
            criticism_count = crit_count,
            criticism_pct   = crit_pct,
            matched_trigger = (
                triggers[best_idx] if best_sim >= CONTROVERSY_MATCH_THRESHOLD
                else f"Unmatched controversy in '{cl.get('label', '')}'"
            ),
            sentiment       = dominant,
            top_comments    = cl.get("top_comments", [])[:3],
        ))

    results.sort(key=lambda c: c.criticism_count, reverse=True)
    return results[:MAX_CONTROVERSIES]


def _compute_top_unanswered(comments: list[dict], cluster_map: dict) -> list[UnansweredQuestion]:
    results = []
    for c in comments:
        cid = c.get("cluster_id", -1)
        cl  = cluster_map.get(cid)
        if not cl or cl.get("cluster_type") != "topic" or _is_fan_cluster(cl):
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


# ── Server-side slot selection ────────────────────────────────────────────────

def _select_video_slots(
    clusters: list[dict],
    gaps:     list[ContentGap],
    n:        int = 5,
) -> list[dict]:
    """
    Pre-select N clusters as video idea slots.
    Server decides WHICH clusters to include — sorted by demand signal.
    LLM decides HOW to describe each (title, why, demand_score, format).

    Priority order:
      1. Content gap clusters first (confirmed unmet demand)
      2. Remaining topic clusters by question_count descending
      Fan clusters excluded from all slots.
    """
    gap_ids = {g.cluster_id for g in gaps}
    slots   = []

    # Priority 1: gap clusters
    for g in gaps:
        if len(slots) >= n:
            break
        bd = next(
            (cl.get("intent_breakdown", {}) for cl in clusters if cl["cluster_id"] == g.cluster_id),
            {},
        )
        q_count = bd.get("question", {}).get("count", 0)
        slots.append({
            "slot_num":     len(slots) + 1,
            "label":        g.label,
            "comment_count": g.comment_count,
            "question_pct": g.question_pct,
            "question_count": q_count,
            "type":         "content_gap",
            "signal":       f"{g.question_pct:.0f}% questions ({q_count} comments) — topic not covered in video",
        })

    # Priority 2: highest-demand non-gap topic clusters
    remaining = sorted(
        [
            cl for cl in clusters
            if cl.get("cluster_type") == "topic"
            and not _is_fan_cluster(cl)
            and cl["cluster_id"] not in gap_ids
        ],
        key=lambda c: c.get("intent_breakdown", {}).get("question", {}).get("count", 0),
        reverse=True,
    )

    for cl in remaining:
        if len(slots) >= n:
            break
        bd         = cl.get("intent_breakdown", {})
        q_pct      = bd.get("question", {}).get("pct", 0)
        q_count    = bd.get("question", {}).get("count", 0)
        crit_pct   = bd.get("criticism", {}).get("pct", 0)

        if crit_pct > 25 and q_pct < 20:
            slot_type = "controversy"
            signal    = f"{crit_pct:.0f}% criticism — audience challenging claims, clarification/sources needed"
        else:
            slot_type = "high_questions"
            signal    = f"{q_pct:.0f}% questions ({q_count} comments asking HOW or WHICH)"

        slots.append({
            "slot_num":      len(slots) + 1,
            "label":         cl.get("label", ""),
            "comment_count": cl.get("comment_count", 0),
            "question_pct":  round(q_pct, 1),
            "question_count": q_count,
            "type":          slot_type,
            "signal":        signal,
        })

    return slots


# ── Call 1: Strategic layer (70b) ─────────────────────────────────────────────

async def _call_strategic_layer(
    video_slots:    list[dict],
    intent_counts:  dict,
    gaps:           list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies:  list[ControversyHotspot],
    clusters:       list[dict],
    summary:        dict,
    api_key:        str,
    model:          str,
) -> dict:
    total         = sum(intent_counts.values()) or 1
    client        = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    system_prompt = _strategic_system_prompt()
    user_prompt   = _strategic_user_prompt(
        video_slots, intent_counts, total, gaps, misconceptions, controversies, clusters, summary
    )

    try:
        response = await client.chat.completions.create(
            model           = model,
            messages        = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format = {"type": "json_object"},
            max_tokens      = 1800,
            temperature     = 0.2,
        )
        raw  = response.choices[0].message.content
        data = _safe_parse(raw)

        if not data:
            logger.warning("strategic_empty_parse", preview=raw[:200])
            return {}

        # Verify the model used the locked stats — detect hallucination
        sv = data.get("stats_verified", {})
        reported_total = sv.get("total", 0)
        if reported_total and abs(reported_total - total) > max(5, total * 0.02):
            logger.warning(
                "strategic_stats_mismatch",
                expected=total, reported=reported_total,
            )

        result = _extract_strategic(data, expected_total=total)
        logger.info(
            "strategic_layer_done",
            stage=result.get("audience_stage"),
            mood=result.get("audience_mood"),
            ideas=len(result.get("top_video_ideas", [])),
            purchase_signals=len(result.get("purchase_intent_signals", [])),
        )
        return result

    except Exception as exc:
        logger.warning("strategic_layer_failed", error=str(exc))
        return {}


def _strategic_system_prompt() -> str:
    return """\
You are a YouTube content strategist converting audience comment data into creator decisions.

YOUR ONLY JOB: fill the JSON template from the data provided. Do not use external knowledge.

LOCKED DATA RULE: The ╔══ LOCKED DATA ══╗ block contains the exact numbers you must use.
Copy them verbatim into stats_verified and executive_summary. Using different numbers = error.

REASONING RULE: Fill the "reasoning" field FIRST. Commit to your analysis before writing answers.
Your reasoning must cite specific numbers from LOCKED DATA to justify each decision.

EXECUTIVE SUMMARY RULES:
  S1: Cite praise_pct and total from LOCKED DATA. Describe what the majority feels.
  S2: Name the #1 video slot and cite its evidence_count. Explain why it is the top opportunity.
  S3: Name the specific misinformation risk and what could happen if not addressed.
  Max 3 sentences total. Cite only numbers from LOCKED DATA.

AUDIENCE STAGE — reason from intent distribution, pick one:
  "Awareness"      → audience is just discovering the topic (praise dominant, few questions)
  "Interest"       → audience wants protocols/brands (questions 20-30%, "which product")
  "Implementation" → audience is actively trying the advice (questions >25%, "how long", results)
  "Evaluation"     → audience critically assessing claims (criticism >20%)
  "Advocacy"       → audience defending and sharing the creator (praise >50%, low questions)

AUDIENCE MOOD — reason from dominant sentiment + intent, pick one:
  "Excited" | "Curious" | "Confused" | "Skeptical" | "Satisfied"

VIDEO IDEAS — convert EVERY slot in the VIDEO SLOTS section. All slots required.
  title:          specific YouTube title, max 70 chars, copy-pasteable
  demand_score:   0-100 — rank slots relative to each other by (question_count × question_pct)
                  highest demand slot gets 90-100, lowest gets 50-65
  confidence_pct: 0-100 — how specific and consistent are the questions in that slot?
                  many identical questions = high confidence; vague interest = low
  evidence_count: the comment_count from the slot data — copy exactly, do not estimate
  why:            1-2 sentences citing exact numbers from slot data only
  format:         "long_video" | "short" | "live_qa" | "community_post"

PURCHASE INTENT — only include if sample comments clearly ask about buying a specific product.
  Format: "N viewers asking about [specific product/brand]"
  If unclear, omit entirely.

CONTENT SERIES — 4-5 titles forming a learning sequence: foundation → advanced → specific.

RISK ALERTS — source from misconceptions + controversies only. Max 3.
  Format: "N viewers believe [specific wrong belief] — [one-line action]"

Return ONLY valid JSON. No markdown. No explanation outside the JSON."""


def _strategic_user_prompt(
    video_slots:    list[dict],
    intent_counts:  dict,
    total:          int,
    gaps:           list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies:  list[ControversyHotspot],
    clusters:       list[dict],
    summary:        dict,
) -> str:
    overview   = summary.get("overview", "")
    key_claims = summary.get("key_claims", [])[:5]

    # Build intent breakdown lines for locked data block
    sorted_intents = sorted(intent_counts.items(), key=lambda x: -x[1])
    intent_lines   = "\n".join(
        f"  {k:<15}: {v:>5} ({v / total * 100:.1f}%)"
        for k, v in sorted_intents
    )

    # Sample comments from top 2 clusters for purchase intent detection
    top_clusters = sorted(
        [cl for cl in clusters if cl.get("cluster_type") == "topic" and not _is_fan_cluster(cl)],
        key=lambda c: c.get("comment_count", 0), reverse=True,
    )
    sample_lines = []
    for cl in top_clusters[:2]:
        for c in cl.get("top_comments", [])[:3]:
            text = (c.get("text") or "")[:120].replace("\n", " ")
            sample_lines.append(f'  [{c.get("like_count", 0):>5} likes] "{text}"')

    lines = [
        f"VIDEO: {overview}",
        f"KEY CLAIMS: {'; '.join(key_claims)}",
        "",
        "╔══ LOCKED DATA — copy these numbers verbatim, do not change them ══╗",
        f"  total_classified : {total}",
        f"  intent breakdown :",
        intent_lines,
        "╚═════════════════════════════════════════════════════════════════════╝",
        "",
        "VIDEO SLOTS — convert all slots below into video ideas (all required):",
    ]

    for s in video_slots:
        lines.append(
            f"  SLOT {s['slot_num']} → \"{s['label']}\" "
            f"| {s['comment_count']} comments | {s['question_pct']:.0f}% questions "
            f"| evidence_count={s['comment_count']} | type={s['type']}"
            f"\n           signal: {s['signal']}"
        )

    if sample_lines:
        lines += ["", "SAMPLE COMMENTS (for purchase intent detection):"]
        lines += sample_lines

    lines += ["", "FINDINGS SUMMARY:"]
    if gaps:
        lines.append(f"  Gaps       : {', '.join(g.label for g in gaps)}")
    if misconceptions:
        lines.append(f"  Misconceptions : {', '.join(f'{m.cluster_label} ({m.misconception_count})' for m in misconceptions)}")
    if controversies:
        lines.append(f"  Controversies  : {', '.join(f'{c.cluster_label} ({c.criticism_count} critical)' for c in controversies)}")

    lines += [
        "",
        "STEP-BY-STEP (fill reasoning field with this analysis):",
        "  1. Look at intent breakdown. Which intent dominates? What stage does this signal?",
        "  2. Rank the SLOTS by (question_count × question_pct). Which has highest demand?",
        "  3. Scan sample comments for 'which brand', 'where to buy', 'recommend a product'.",
        "  4. Which misconception or controversy poses the highest credibility risk?",
        "  5. Write executive_summary last using ONLY numbers from LOCKED DATA.",
        "",
        "Fill this JSON (all fields required):",
        json.dumps({
            "reasoning": "<step-by-step analysis citing numbers from locked data>",
            "stats_verified": {
                "total": "<copy total_classified from locked data>",
                "dominant_intent": "<intent name>",
                "dominant_intent_pct": "<pct from locked data>",
            },
            "executive_summary": "<3 sentences using locked data numbers>",
            "audience_stage": "<one of: Awareness|Interest|Implementation|Evaluation|Advocacy>",
            "audience_mood": "<one of: Excited|Curious|Confused|Skeptical|Satisfied>",
            "top_video_ideas": [
                {
                    "rank": 1,
                    "title": "<specific YouTube title>",
                    "demand_score": "<0-100>",
                    "confidence_pct": "<0-100>",
                    "why": "<1-2 sentences citing slot data>",
                    "evidence_count": "<copy comment_count from slot>",
                    "format": "<long_video|short|live_qa|community_post>",
                },
                {"rank": 2, "title": "", "demand_score": 0, "confidence_pct": 0, "why": "", "evidence_count": 0, "format": ""},
                {"rank": 3, "title": "", "demand_score": 0, "confidence_pct": 0, "why": "", "evidence_count": 0, "format": ""},
                {"rank": 4, "title": "", "demand_score": 0, "confidence_pct": 0, "why": "", "evidence_count": 0, "format": ""},
                {"rank": 5, "title": "", "demand_score": 0, "confidence_pct": 0, "why": "", "evidence_count": 0, "format": ""},
            ],
            "purchase_intent_signals": ["<only if clear buying intent in sample comments>"],
            "content_series": ["<title 1>", "<title 2>", "<title 3>", "<title 4>", "<title 5>"],
            "risk_alerts": ["<N viewers believe X — action>"],
        }, indent=2),
    ]

    return "\n".join(lines)


# ── Call 2: Per-item enrichment (8b) ─────────────────────────────────────────

async def _call_per_item_enrichment(
    gaps:           list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies:  list[ControversyHotspot],
    summary:        dict,
    api_key:        str,
    model:          str,
) -> None:
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    try:
        response = await client.chat.completions.create(
            model           = model,
            messages        = [
                {"role": "system", "content": _enrichment_system_prompt()},
                {"role": "user",   "content": _enrichment_user_prompt(
                    gaps, misconceptions, controversies, summary
                )},
            ],
            response_format = {"type": "json_object"},
            max_tokens      = 1500,
            temperature     = 0.2,
        )
        data = _safe_parse(response.choices[0].message.content)
        _apply_enrichment(data, gaps, misconceptions, controversies)
        logger.info("per_item_enrichment_done")
    except Exception as exc:
        logger.warning("per_item_enrichment_failed", error=str(exc))
        _apply_fallback_labels(gaps, misconceptions, controversies)


def _enrichment_system_prompt() -> str:
    return """\
You are a YouTube content strategist. Write one creator action for EVERY finding below.

FIELD RULES:
  what_to_do:     One sentence — the specific action. Start with a verb.
  why:            Cite exact comment count AND quote one example comment. No vague claims.
  suggested_hook: Copy-pasteable text. See type rules below.
  urgency:        "high" (≥100 comments or direct credibility risk) | "medium" (20-99) | "low" (<20)
  impact_type:    "new_video" | "pin_comment" | "update_description" | "community_post"

SUGGESTED_HOOK RULES BY TYPE:
  content_gap    → A specific YouTube video title (max 70 chars). Copy-pasteable as-is.
  misconception  → A pinned comment correction. Start with "CLARIFICATION:".
                   ONLY use what is stated in KEY CLAIMS — do not add external science.
                   If no KEY CLAIM matches → write: "CLARIFICATION: As explained in this video,
                   [paraphrase the video's position on this topic]. Consult a healthcare professional."
  controversy    → Description text for the video description. Do NOT name specific journals.
                   Write: "For sources on [topic], see the links in the video description."
                   OR: "Peer-reviewed research on [topic] supports the position in this video."

HARD RULES:
  1. Enrich EVERY item — do not skip any cluster_id.
  2. NEVER cite specific journal names, study titles, or publication years.
  3. NEVER state scientific facts not present in KEY CLAIMS.
  4. urgency must match comment count: ≥100=high, 20-99=medium, <20=low. No exceptions.
  5. what_to_do must be a single sentence starting with a verb.

EXAMPLES:

Content gap — 50 comments, 76% questions:
{
  "cluster_id": 8,
  "what_to_do": "Create a dedicated video covering astigmatism and whether the remedies in this video apply",
  "why": "50 viewers asked about astigmatism with no answer from the video. Top comment (60 likes): 'Please can you make a video on causes and fixes for astigmatism'",
  "suggested_hook": "Can You Fix Astigmatism Naturally? (What the Research Shows)",
  "urgency": "medium",
  "impact_type": "new_video"
}

Misconception — 11 comments, related_claim exists:
{
  "cluster_id": 3,
  "what_to_do": "Pin a comment clarifying what NAC eye drops actually do based on the video's claims",
  "why": "11 viewers overstated NAC results. Top comment (38 likes): 'NAC is quite effective — my vision improved by 1.5 lines'",
  "suggested_hook": "CLARIFICATION: As explained in this video, NAC eye drops help slow cataract progression — they do not permanently reverse advanced cataracts. Please consult an ophthalmologist for advanced cases.",
  "urgency": "medium",
  "impact_type": "pin_comment"
}

Controversy — 49 critical comments, 73% of cluster:
{
  "cluster_id": 7,
  "what_to_do": "Add a sources section to the video description addressing LED light and eye health",
  "why": "49 critical comments (73% of cluster) challenge LED safety claims. Top comment (805 likes): 'LED car headlights literally blind me for a few seconds.'",
  "suggested_hook": "For peer-reviewed sources on LED light and eye health, see the links in the video description.",
  "urgency": "high",
  "impact_type": "update_description"
}

Return ONLY valid JSON:
{
  "content_gaps":         [{"cluster_id": <int>, "what_to_do": "", "why": "", "suggested_hook": "", "urgency": "", "impact_type": ""}],
  "misconceptions":       [{"cluster_id": <int>, "what_to_do": "", "why": "", "suggested_hook": "", "urgency": "", "impact_type": ""}],
  "controversy_hotspots": [{"cluster_id": <int>, "what_to_do": "", "why": "", "suggested_hook": "", "urgency": "", "impact_type": ""}]
}"""


def _enrichment_user_prompt(
    gaps:           list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies:  list[ControversyHotspot],
    summary:        dict,
) -> str:
    overview   = summary.get("overview", "")
    key_claims = summary.get("key_claims", [])[:8]

    lines = [
        f"VIDEO: {overview}",
        "",
        "KEY CLAIMS (anchor all hooks to these — do not add external knowledge):",
    ]
    for i, claim in enumerate(key_claims, 1):
        lines.append(f"  {i}. {claim}")

    lines += ["", "Enrich EVERY item listed below. Do not skip any.", ""]

    if gaps:
        lines.append("=== CONTENT GAPS ===")
        for g in gaps:
            sample = g.top_comments[0]["text"][:120] if g.top_comments else ""
            likes  = g.top_comments[0].get("like_count", 0) if g.top_comments else 0
            lines.append(
                f'cluster_id={g.cluster_id} | "{g.label}" | '
                f'{g.comment_count} comments | {g.question_pct}% questions\n'
                f'  top comment ({likes} likes): "{sample}"'
            )
        lines.append("")

    if misconceptions:
        lines.append("=== MISCONCEPTIONS ===")
        for m in misconceptions:
            top_text  = m.top_comments[0]["text"][:120] if m.top_comments else ""
            top_likes = m.top_comments[0].get("like_count", 0) if m.top_comments else 0
            claim_str = f'\n  matching KEY CLAIM: "{m.related_claim}"' if m.related_claim else "\n  (no matching key claim — use generic clarification)"
            lines.append(
                f'cluster_id={m.cluster_id} | "{m.cluster_label}" | '
                f'{m.misconception_count} misconception comments{claim_str}\n'
                f'  top comment ({top_likes} likes): "{top_text}"'
            )
        lines.append("")

    if controversies:
        lines.append("=== CONTROVERSY HOTSPOTS ===")
        for c in controversies:
            top_text  = c.top_comments[0]["text"][:120] if c.top_comments else ""
            top_likes = c.top_comments[0].get("like_count", 0) if c.top_comments else 0
            lines.append(
                f'cluster_id={c.cluster_id} | "{c.cluster_label}" | '
                f'{c.criticism_count} critical comments ({c.criticism_pct:.0f}%) | '
                f'sentiment={c.sentiment}\n'
                f'  trigger: "{c.matched_trigger}"\n'
                f'  top comment ({top_likes} likes): "{top_text}"'
            )
        lines.append("")

    lines.append("Write the JSON now. Include every cluster_id listed above.")
    return "\n".join(lines)


# ── Parsing and validation ────────────────────────────────────────────────────

def _safe_parse(raw: str) -> dict:
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {}


def _extract_strategic(data: dict, expected_total: int = 0) -> dict:
    video_ideas = []
    for i, v in enumerate(data.get("top_video_ideas", [])[:MAX_VIDEO_IDEAS]):
        if not isinstance(v, dict):
            continue
        title = str(v.get("title", "")).strip()
        if not title or title.startswith("<"):  # skip unfilled template slots
            continue
        try:
            fmt = str(v.get("format", "long_video"))
            video_ideas.append(VideoIdea(
                rank           = int(v.get("rank", i + 1)),
                title          = title[:120],
                demand_score   = min(100, max(0, int(v.get("demand_score", 50)))),
                confidence_pct = min(100, max(0, int(v.get("confidence_pct", 50)))),
                why            = str(v.get("why", "")).strip(),
                evidence_count = max(0, int(v.get("evidence_count", 0))),
                format         = fmt if fmt in VALID_FORMATS else "long_video",
            ))
        except (TypeError, ValueError):
            continue

    stage = str(data.get("audience_stage", "")).strip()
    mood  = str(data.get("audience_mood",  "")).strip()

    return {
        "executive_summary":       str(data.get("executive_summary", "")).strip(),
        "audience_stage":          stage if stage in VALID_STAGES else "",
        "audience_mood":           mood  if mood  in VALID_MOODS  else "",
        "top_video_ideas":         video_ideas,
        "purchase_intent_signals": [str(s).strip() for s in data.get("purchase_intent_signals", []) if s and not str(s).startswith("<")],
        "content_series":          [str(s).strip() for s in data.get("content_series", []) if s and not str(s).startswith("<")],
        "risk_alerts":             [str(s).strip() for s in data.get("risk_alerts", []) if s and not str(s).startswith("<")],
    }


def _apply_enrichment(
    data:           dict,
    gaps:           list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies:  list[ControversyHotspot],
) -> None:
    def _apply(items, key, default_impact):
        enriched = {
            str(item["cluster_id"]): item
            for item in data.get(key, [])
            if isinstance(item, dict) and "cluster_id" in item
        }
        for obj in items:
            e = enriched.get(str(obj.cluster_id), {})
            obj.what_to_do     = str(e.get("what_to_do", "")).strip()
            obj.why            = str(e.get("why", "")).strip()
            obj.suggested_hook = str(e.get("suggested_hook", "")).strip()
            obj.urgency        = (
                e.get("urgency") if e.get("urgency") in ("high", "medium", "low")
                else "medium"
            )
            obj.impact_type    = (
                e.get("impact_type") if e.get("impact_type") in
                ("new_video", "pin_comment", "update_description", "community_post")
                else default_impact
            )

    _apply(gaps,           "content_gaps",        "new_video")
    _apply(misconceptions, "misconceptions",       "pin_comment")
    _apply(controversies,  "controversy_hotspots", "update_description")


def _apply_fallback_labels(
    gaps:           list[ContentGap],
    misconceptions: list[MisconceptionItem],
    controversies:  list[ControversyHotspot],
) -> None:
    for g in gaps:
        if not g.what_to_do:
            g.what_to_do  = f"Create content covering '{g.label}'"
            g.urgency     = "high" if g.comment_count >= 100 else "medium"
            g.impact_type = "new_video"
    for m in misconceptions:
        if not m.what_to_do:
            m.what_to_do  = f"Pin a clarification about '{m.cluster_label}'"
            m.urgency     = "high" if m.misconception_count >= 20 else "medium"
            m.impact_type = "pin_comment"
    for c in controversies:
        if not c.what_to_do:
            c.what_to_do  = f"Add sources for claims in '{c.cluster_label}'"
            c.urgency     = "high" if c.criticism_count >= 50 else "medium"
            c.impact_type = "update_description"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slim(c: dict) -> dict:
    return {
        "comment_id": c.get("comment_id", ""),
        "text":       (c.get("text") or "")[:200],
        "like_count": c.get("like_count") or 0,
    }
