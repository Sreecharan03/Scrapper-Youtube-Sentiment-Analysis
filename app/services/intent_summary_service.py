"""
app/services/intent_summary_service.py
========================================
Phase 3E-pre: Per-intent audience summary using Claude Haiku.

For each of 6 meaningful intent categories (question, praise, criticism,
confusion, misconception, request) generates a 2-3 sentence summary that
tells the creator specifically what their audience is saying — citing
topics, clusters, and real examples.

spam / off_topic → static count-only lines, no LLM tokens spent.

ONE Claude Haiku call returns all summaries + an overall summary as JSON.

INPUT:
  - comment_repo  → per-intent top comments (by like_count) + counts
  - cluster_map   → which clusters dominate each intent (richer context)
  - summary doc   → video overview for Claude system context

OUTPUT (stored in MongoDB):
  {
    overall_summary: str,
    intent_summaries: {
      question:     {summary, count},
      praise:       {summary, count},
      ...
      spam:         {summary, count},   ← static, no LLM
      off_topic:    {summary, count},   ← static, no LLM
    }
  }
"""

import json

import anthropic

from app.core.logging import get_logger

logger = get_logger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

LLM_INTENTS    = ["question", "praise", "criticism", "confusion", "misconception", "request"]
STATIC_INTENTS = ["spam", "off_topic"]
TOP_N_COMMENTS = 8   # per intent, fed to Claude
TOP_N_CLUSTERS = 3   # top clusters per intent shown to Claude


class IntentSummaryService:

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    async def generate(
        self,
        video_id:     str,
        intent_counts: dict,          # {intent: count}
        top_comments:  dict,          # {intent: [comment_dicts]}
        clusters:      list[dict],    # all cluster docs
        video_summary: dict,          # summary doc from 3A
    ) -> dict:

        # ── Static summaries for noise intents ───────────────────────────
        static: dict[str, dict] = {}
        for intent in STATIC_INTENTS:
            count = intent_counts.get(intent, 0)
            label = "spam" if intent == "spam" else "off-topic"
            static[intent] = {
                "summary": f"{count} {label} comments were detected and excluded from analysis.",
                "count":   count,
            }

        # ── Build cluster context per LLM intent ─────────────────────────
        cluster_context = _build_cluster_context(clusters, LLM_INTENTS, TOP_N_CLUSTERS)

        # ── Call Claude Haiku ─────────────────────────────────────────────
        overview  = video_summary.get("overview", "")
        llm_result = self._call_claude(
            video_id, overview, intent_counts, top_comments, cluster_context
        )

        # ── Merge ─────────────────────────────────────────────────────────
        intent_summaries: dict[str, dict] = {}
        for intent in LLM_INTENTS:
            intent_summaries[intent] = {
                "summary": llm_result.get(intent, ""),
                "count":   intent_counts.get(intent, 0),
            }
        for intent in STATIC_INTENTS:
            intent_summaries[intent] = static[intent]

        return {
            "overall_summary":  llm_result.get("overall", ""),
            "intent_summaries": intent_summaries,
        }

    def _call_claude(
        self,
        video_id:        str,
        overview:        str,
        intent_counts:   dict,
        top_comments:    dict,
        cluster_context: dict,
    ) -> dict:

        system_prompt = _build_system_prompt()
        user_prompt   = _build_user_prompt(
            overview, intent_counts, top_comments, cluster_context
        )

        try:
            response = self._client.messages.create(
                model      = HAIKU_MODEL,
                max_tokens = 2048,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            return _extract_json(raw)
        except Exception as exc:
            logger.warning("intent_summary_claude_failed", video_id=video_id, error=str(exc))
            return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_cluster_context(
    clusters:       list[dict],
    intents:        list[str],
    top_n:          int,
) -> dict[str, list[dict]]:
    """
    For each intent, find the top_n clusters ranked by that intent's comment count.
    Returns {intent: [{label, count, pct}]}
    """
    context: dict[str, list[dict]] = {intent: [] for intent in intents}

    for intent in intents:
        ranked = []
        for cl in clusters:
            bd    = cl.get("intent_breakdown", {})
            entry = bd.get(intent, {})
            count = entry.get("count", 0)
            pct   = entry.get("pct", 0)
            if count > 0:
                ranked.append({
                    "label": cl.get("label", ""),
                    "count": count,
                    "pct":   pct,
                })
        ranked.sort(key=lambda x: x["count"], reverse=True)
        context[intent] = ranked[:top_n]

    return context


def _build_system_prompt() -> str:
    return """\
You are a senior YouTube audience intelligence analyst. Your job is to read \
comment cluster data for a YouTube video and write sharp, specific, creator-ready \
summaries for each intent category — summaries the creator can act on TODAY.

QUALITY STANDARD — every summary must pass this test:
"Could a creator read this in 10 seconds and know exactly what their audience is \
thinking AND what to do about it?"

❌ BAD (vague, useless):
"Many viewers had questions about various topics in the video."

✅ GOOD (specific + evidence + action):
"856 viewers are asking about eye health, with the biggest demand in NAC eye drop \
protocols (130 comments, 58% questions) and astigmatism treatments (50 comments, 76% \
questions). The top-liked question (66 likes) asks how long castor oil takes to show \
results on cataracts. Pin a FAQ comment covering NAC dosing, castor oil timelines, \
and astigmatism to address 40% of all questions in one post."

INTENT-SPECIFIC WRITING RULES:
- question:      Focus on WHAT they are asking (topic + volume). Which cluster drives most questions?
- praise:        Focus on WHAT they specifically loved (not just that they liked it). Quote the feeling.
- criticism:     Name the SPECIFIC claim being challenged. What exactly are they pushing back on?
- confusion:     Describe WHERE in the content they got lost (concept, term, or claim).
- misconception: State the wrong belief clearly. Which video claim is being misunderstood?
- request:       Name the exact content format + topic they want (Short, follow-up video, live Q&A).
- overall:       3-sentence arc — what is the dominant feeling, what is the #1 need, what is at risk?

HARD RULES:
1. Each summary: exactly 2-3 sentences. No more.
2. ALWAYS cite real numbers (counts, percentages, like counts).
3. ALWAYS end with one concrete creator action.
4. NEVER say "many viewers", "some people", "various topics" — be specific.
5. NEVER repeat the intent category name as the first word.

FEW-SHOT EXAMPLES (from a fitness video about protein intake):

Input for QUESTION (720 comments, top cluster "Protein Timing" with 312 questions):
Output: "720 viewers are asking about protein, with 312 concentrated in one cluster \
around pre vs post-workout timing — the single most-asked question in the video. The \
top question (89 likes) asks whether plant protein counts the same as animal protein \
for muscle synthesis. A 60-second Short titled 'Plant vs Animal Protein: What Actually \
Counts' would directly answer 40% of all unanswered questions."

Input for PRAISE (1,200 comments, top cluster "Explanation Clarity" 680 praise comments):
Output: "1,200 viewers praised the video, with 680 specifically calling out how clearly \
compound movements were explained — the top comment (2,100 likes) says 'Finally someone \
explained sets vs reps without making me feel stupid.' The praise is concentrated around \
the beginner-friendly breakdown at the 3-minute mark, not the advanced content. Clip that \
segment into a Short with the caption 'The only sets and reps explanation you need' to \
maximize reach with new audiences."

Input for MISCONCEPTION (340 comments, cluster "Protein Limit Myths"):
Output: "340 viewers believe the body can only absorb 30g of protein per meal — a myth \
the video did not debunk directly. The top misconception comment (45 likes) says 'I \
always stop at 30g per meal because that's the max your body absorbs at once.' Pin a \
comment citing the 2023 Norton et al. study showing total daily protein matters more \
than per-meal limits to correct this at the source."

Return ONLY a valid JSON object — no markdown, no explanation — with exactly these keys:
overall, question, praise, criticism, confusion, misconception, request"""


def _build_user_prompt(
    overview:        str,
    intent_counts:   dict,
    top_comments:    dict,
    cluster_context: dict,
) -> str:
    total = sum(intent_counts.get(i, 0) for i in LLM_INTENTS)
    lines = [
        f"VIDEO OVERVIEW: {overview}",
        f"TOTAL CLASSIFIED COMMENTS: {total}",
        f"INTENT BREAKDOWN: " + ", ".join(
            f"{i}={intent_counts.get(i,0)} ({round(intent_counts.get(i,0)/max(total,1)*100,1)}%)"
            for i in LLM_INTENTS
        ),
        "",
        "Think step by step for each intent: What is the audience specifically saying? "
        "Which topic cluster drives this the most? What would help the creator most?",
        "Then write the summary.",
        "",
    ]

    for intent in LLM_INTENTS:
        count    = intent_counts.get(intent, 0)
        pct      = round(count / max(total, 1) * 100, 1)
        clusters = cluster_context.get(intent, [])
        comments = top_comments.get(intent, [])

        lines.append(f"=== {intent.upper()} — {count} comments ({pct}%) ===")

        if clusters:
            lines.append("Where it concentrates:")
            for c in clusters:
                lines.append(
                    f'  · "{c["label"]}": {c["count"]} {intent} comments ({c["pct"]:.0f}% of that cluster)'
                )

        if comments:
            lines.append("Top comments by likes:")
            for c in comments[:6]:
                text  = (c.get("text") or "").replace("\n", " ")[:180]
                likes = c.get("like_count") or 0
                lines.append(f'  [{likes} likes] "{text}"')

        lines.append("")

        lines.append("")

    lines.append(
        "Write the JSON summaries now. "
        "Keys: overall, question, praise, criticism, confusion, misconception, request"
    )
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {}
