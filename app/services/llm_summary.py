"""
app/services/llm_summary.py
============================
LLM-powered transcript summary service using Claude Haiku.

PROMPT STRATEGY — THREE LAYERS:

  Layer 1 — System prompt (cached):
    Persona + strict rules + full JSON schema definition.
    Cached via Anthropic prompt caching — paid once, reused across all videos.

  Layer 2 — Few-shot examples (cached inside system):
    Two complete input→output examples (competition + educational).
    Teaches the model what "specific" vs "generic" looks like for every field.
    Teaches timestamp accuracy, Gen Z audience calibration, named entity
    completeness, and the difference between a humor moment and casual chat.

  Layer 3 — Self-critique loop (2 calls total):
    Call 1: Generate draft summary from transcript.
    Call 2: Feed draft back → model lists specific flaws → produces refined output.
    Transcript is cached for Call 2 → ~10x cheaper than re-sending it.

WHY SELF-CRITIQUE:
  Without critique, Haiku tends to:
    - Write generic controversy_triggers ("was this real?") even when the
      transcript has no staging hints
    - Miss named entities that appear only once
    - Misclassify skull emoji context without checking emotional_arc
    - Write key_claims as paraphrases instead of actual transcript statements
  The critique call forces explicit issue identification before the final output.

CACHING ECONOMICS (Haiku):
  Transcript ~6,000 tokens. Without cache: $0.80/M × 6k = $0.0048 per call.
  With cache read: $0.08/M × 6k = $0.00048 per call (10x cheaper).
  System prompt ~3,000 tokens also cached.
  Total per video (2 calls): ~$0.012 — negligible at any scale.
"""

import json
import re
import time
from typing import Optional

import anthropic

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ── Prompts ───────────────────────────────────────────────────────────────

# NOTE: This prompt is sent as the system block with cache_control=ephemeral.
# It is paid once per 5-minute cache TTL window, not per request.
SYSTEM_PROMPT = """You are an expert YouTube video intelligence analyst. Your analyses power a downstream \
comment sentiment classification system, so accuracy and specificity are critical. Vague or generic output \
directly degrades the quality of sentiment analysis for 300,000+ comments.

## Your Role
Extract structured metadata from YouTube video transcripts. Every field you produce will be used to:
- Correct emoji misclassification (💀 on a funny moment ≠ negative)
- Separate "opinion about a person" from "opinion about the video"
- Identify controversy clusters before sentiment scoring
- Detect broken-promise complaints as a distinct sentiment type
- Calibrate the sentiment model for the specific audience demographic

## Non-Negotiable Rules
1. GROUND every field in direct transcript evidence — never invent or assume
2. controversy_triggers must be SPECIFIC to THIS video's claims — not generic YouTube tropes
3. humor_moments must be moments where something clearly funny HAPPENS — not just casual conversation
4. key_claims must be near-verbatim statements from the transcript — not your paraphrase
5. named_entities must include EVERY named person, brand, product, and prize mentioned
6. If you cannot find transcript evidence for a field → use null or empty array []
7. Output ONLY valid JSON — no markdown fences, no preamble, no commentary

## Output Schema
{
  "overview": "2-3 factual sentences describing what happens in the video",
  "key_topics": [
    {"label": "string", "start_ms": int, "end_ms": int, "description": "string"}
  ],
  "key_claims": ["verbatim or near-verbatim claims from the transcript"],
  "emotional_arc": [
    {"start_ms": int, "end_ms": int, "emotion": "tense|exciting|funny|sad|inspiring|scary|neutral", "description": "string"}
  ],
  "named_entities": {
    "people": ["every named person mentioned"],
    "brands": ["every brand, product, show, or IP mentioned"],
    "prizes": ["every monetary amount or prize mentioned"]
  },
  "controversy_triggers": ["specific debatable claims or moments — must have transcript evidence"],
  "video_promises": ["explicit commitments the host makes to the viewer"],
  "humor_moments": [
    {"start_ms": int, "end_ms": int, "description": "what specifically is funny"}
  ],
  "audience_signals": {
    "age_group": "kids|teens|young_adults|adults|mixed",
    "language_style": "formal|casual|casual_gen_z|academic|mixed",
    "engagement_type": "competitive_entertainment|education|vlog|tutorial|debate|reaction|other",
    "likely_emotions": ["emotions viewers will primarily feel while watching"]
  },
  "content_warnings": ["ethical or sensitive aspects that may trigger complaints unrelated to video quality"],
  "tone": "string describing overall tone",
  "content_type": "string category"
}

---

## Few-Shot Example 1 — Competition / Entertainment

INPUT TRANSCRIPT (excerpt, ~800 words):
"Welcome to the final challenge. 100 people signed up for this. Only one walks away with \
50 thousand dollars. Here are the rules: stay inside this 10-foot circle. Last person standing \
wins. Ready? Go. [hour 1] Everyone's still in. People are stretching, chatting, getting comfortable. \
Jake keeps making everyone laugh by doing impressions of the host. [hour 4] 40 people left. It is \
genuinely getting hard to stand here. One girl just started crying and hugged her friend before \
stepping out — that was emotional. [hour 7] Down to 12 people. The circle is getting tense. \
Nobody's talking anymore. Pure focus. Guy in position 8 is swaying — he's going down. Yep, he's out. \
[hour 9] Final 3. This is insane. Sarah, Mike and contestant 44. 44 just told Sarah he'd split the \
money if she steps out. She declined. Respect. [hour 10, final moments] Mike collapses. Final 2: \
Sarah and 44. They've agreed to keep going. 44 steps out. SARAH WINS. 50 thousand dollars goes to \
Sarah. Subscribe to win a signed merch pack this week."

EXPECTED OUTPUT:
{
  "overview": "100 contestants stand in a 10-foot circle for up to 10 hours competing for $50,000. Players are gradually eliminated as fatigue sets in, culminating in a final two-person standoff won by a contestant named Sarah.",
  "key_topics": [
    {"label": "Opening rules and setup", "start_ms": 0, "end_ms": 600000, "description": "Host explains competition, 100 players begin"},
    {"label": "Early hours — social phase", "start_ms": 600000, "end_ms": 14400000, "description": "Players comfortable, Jake doing impressions"},
    {"label": "Mid-game attrition", "start_ms": 14400000, "end_ms": 25200000, "description": "40 players remain, emotional exits begin"},
    {"label": "Final tension phase", "start_ms": 25200000, "end_ms": 32400000, "description": "12 players, silence, pure focus"},
    {"label": "Final 3 — deal offer", "start_ms": 32400000, "end_ms": 36000000, "description": "Contestant 44 offers Sarah a deal; she refuses"},
    {"label": "Winner announced", "start_ms": 36000000, "end_ms": 36600000, "description": "Sarah wins $50,000"}
  ],
  "key_claims": [
    "Last person standing inside the 10-foot circle wins $50,000",
    "100 people competed",
    "Subscribe to win a signed merch pack this week"
  ],
  "emotional_arc": [
    {"start_ms": 0, "end_ms": 600000, "emotion": "exciting", "description": "High-energy intro, stakes established"},
    {"start_ms": 600000, "end_ms": 14400000, "emotion": "funny", "description": "Jake doing impressions, relaxed social energy"},
    {"start_ms": 14400000, "end_ms": 25200000, "emotion": "sad", "description": "Emotional exits, girl crying before leaving"},
    {"start_ms": 25200000, "end_ms": 32400000, "emotion": "tense", "description": "Silent focus, swaying players, high stakes"},
    {"start_ms": 32400000, "end_ms": 36000000, "emotion": "tense", "description": "Deal offer, Sarah refuses — moral moment"},
    {"start_ms": 36000000, "end_ms": 36600000, "emotion": "exciting", "description": "Winner revealed, celebration"}
  ],
  "named_entities": {
    "people": ["Sarah", "Mike", "Jake", "contestant 44"],
    "brands": [],
    "prizes": ["$50,000", "signed merch pack"]
  },
  "controversy_triggers": [
    "Contestant 44 offered Sarah money to step out — was that a legitimate strategy or manipulation?",
    "Was the circle monitored fairly — could contestants cheat by partially stepping out?"
  ],
  "video_promises": [
    "Last person standing wins $50,000",
    "Subscribe to win a signed merch pack this week"
  ],
  "humor_moments": [
    {"start_ms": 3600000, "end_ms": 7200000, "description": "Jake doing impressions of the host — contestants laughing"}
  ],
  "audience_signals": {
    "age_group": "young_adults",
    "language_style": "casual",
    "engagement_type": "competitive_entertainment",
    "likely_emotions": ["excitement", "suspense", "admiration", "humor"]
  },
  "content_warnings": [],
  "tone": "high-energy, suspenseful, emotionally varied",
  "content_type": "endurance_challenge"
}

---

## Few-Shot Example 2 — Educational / Explainer

INPUT TRANSCRIPT (excerpt, ~800 words):
"Today we're asking: how much sleep do you actually need? The CDC says adults need 7 to 9 hours. \
But here's what most people get wrong — it's not just total hours, it's sleep quality. \
Dr. Matthew Walker, author of Why We Sleep, argues that consistently getting under 6 hours \
raises your risk of Alzheimer's by 33 percent — that's a claim a lot of people push back on \
because some studies don't replicate it. We also cover the 90-minute sleep cycle rule. Your brain \
cycles through light sleep, deep sleep and REM roughly every 90 minutes. So if you wake up mid-cycle \
you feel groggy — called sleep inertia. Setting your alarm to complete cycles, so 6 hours or 7.5 hours, \
is better than a random 8. That said, some people are genuine short sleepers — a rare mutation \
called BHLHE41 lets about 3 percent of people thrive on 6 hours. Most people who claim to be fine \
on less sleep are actually just adapted to feeling impaired. The research on this is clear. \
Our sponsor today is Helix Sleep — use code SLEEP20 for 20 percent off."

EXPECTED OUTPUT:
{
  "overview": "An educational deep-dive into sleep science covering the CDC's 7-9 hour recommendation, Dr. Matthew Walker's Alzheimer's risk claim, the 90-minute sleep cycle rule, sleep inertia, and the rare genetic short-sleeper mutation BHLHE41. Sponsored by Helix Sleep.",
  "key_topics": [
    {"label": "CDC sleep recommendation", "start_ms": 0, "end_ms": 120000, "description": "7-9 hours for adults"},
    {"label": "Walker's Alzheimer's claim", "start_ms": 120000, "end_ms": 300000, "description": "Under 6 hours raises Alzheimer's risk 33% — contested"},
    {"label": "90-minute sleep cycles", "start_ms": 300000, "end_ms": 480000, "description": "Light, deep, REM cycle — alarm timing strategy"},
    {"label": "Sleep inertia", "start_ms": 480000, "end_ms": 540000, "description": "Grogginess from waking mid-cycle"},
    {"label": "Short sleeper mutation", "start_ms": 540000, "end_ms": 660000, "description": "BHLHE41 affects ~3% of people — most '6-hour people' are not this"}
  ],
  "key_claims": [
    "CDC says adults need 7 to 9 hours of sleep",
    "Dr. Matthew Walker argues consistently getting under 6 hours raises Alzheimer's risk by 33 percent",
    "Some studies do not replicate Walker's Alzheimer's finding",
    "The brain cycles through light, deep, and REM sleep roughly every 90 minutes",
    "About 3 percent of people carry the BHLHE41 mutation and genuinely thrive on 6 hours",
    "Most people who claim to be fine on less sleep are adapted to feeling impaired"
  ],
  "emotional_arc": [
    {"start_ms": 0, "end_ms": 300000, "emotion": "neutral", "description": "Factual delivery of CDC guidelines"},
    {"start_ms": 300000, "end_ms": 480000, "emotion": "tense", "description": "Alzheimer's risk — alarming statistic followed by immediate caveat"},
    {"start_ms": 480000, "end_ms": 660000, "emotion": "neutral", "description": "Practical advice on sleep cycles and mutation — informational"}
  ],
  "named_entities": {
    "people": ["Dr. Matthew Walker"],
    "brands": ["Helix Sleep", "CDC"],
    "prizes": []
  },
  "controversy_triggers": [
    "Walker's 33% Alzheimer's risk claim is disputed — some studies don't replicate it",
    "The claim that most short sleepers are 'just adapted to feeling impaired' may feel dismissive to viewers"
  ],
  "video_promises": [
    "Helix Sleep discount: code SLEEP20 for 20% off"
  ],
  "humor_moments": [],
  "audience_signals": {
    "age_group": "adults",
    "language_style": "casual",
    "engagement_type": "education",
    "likely_emotions": ["curiosity", "concern", "motivation"]
  },
  "content_warnings": [
    "Alzheimer's risk statistics may cause health anxiety"
  ],
  "tone": "informative, evidence-based, occasionally alarming",
  "content_type": "health_education"
}"""


DRAFT_TASK_PROMPT = """\
Analyse the following video transcript and produce a structured JSON summary.

Apply every rule from your instructions. Be specific — generic output is a failure.
Output ONLY the JSON object. No markdown, no explanation.

<transcript>
{transcript}
</transcript>"""


CRITIQUE_PROMPT = """\
You produced this draft summary for the transcript above:

<draft_summary>
{draft}
</draft_summary>

Now rigorously self-critique it using this exact checklist:

SPECIFICITY: Are ALL controversy_triggers directly evidenced in this transcript? \
Mark any that are generic assumptions not supported by transcript text.

COMPLETENESS: Re-read the transcript. List any named people, brands, prizes, or \
explicit numbers that are MISSING from named_entities.

TIMESTAMP COVERAGE: The video is {duration_secs:.0f} seconds ({duration_ms} ms) long. \
Do your emotional_arc and key_topics timestamps cover the full duration without gaps? \
Do the end timestamps of the last entries match or approach {duration_ms}?

HUMOR PRECISION: Are humor_moments genuinely funny incidents, or just casual chat? \
List any that are misclassified.

CLAIM ACCURACY: Are key_claims near-verbatim from the transcript? \
List any that are your paraphrase rather than actual statements.

AUDIENCE CALIBRATION: Does audience_signals correctly reflect the language style \
and energy of this specific video?

Output your response in exactly this format:

<critique>
SPECIFICITY: [your findings]
COMPLETENESS: [your findings]
TIMESTAMPS: [your findings]
HUMOR: [your findings]
CLAIMS: [your findings]
AUDIENCE: [your findings]
OVERALL_SEVERITY: low|medium|high
</critique>
<refined_summary>
[The complete corrected JSON, fixing every issue identified above]
</refined_summary>"""


# ── JSON extraction (robust) ───────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Extract and parse a JSON object from LLM output.
    Tries in order:
      1. Direct parse (model followed instructions)
      2. Extract from <refined_summary> tags
      3. Extract largest {...} block
      4. Strip markdown fences then parse
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Inside <refined_summary> tags
    m = re.search(r"<refined_summary>\s*(\{.*?\})\s*</refined_summary>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First { to last } — most reliable for JSON inside markdown fences
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 4. Strip markdown fences then try again
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"\s*```", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response. "
        f"First 400 chars: {text[:400]!r}"
    )


def _extract_critique(text: str) -> str:
    """Pull the <critique>...</critique> block out of the critique response."""
    m = re.search(r"<critique>(.*?)</critique>", text, re.DOTALL)
    return m.group(1).strip() if m else "(no critique block found)"


# ── Main service ──────────────────────────────────────────────────────────

class SummaryService:
    """
    Generates a structured video summary using Claude Haiku with:
      - Prompt caching (transcript cached across both calls)
      - Few-shot examples in the system prompt
      - Self-critique loop (draft → critique → refined output)
    """

    def __init__(self, api_key: str, model: str = HAIKU_MODEL) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def generate(
        self,
        transcript_text: str,
        duration_secs:   float = 0.0,
        video_title:     Optional[str] = None,
    ) -> dict:
        """
        Full two-call pipeline:
          1. Generate draft summary
          2. Self-critique + produce refined summary

        Returns the refined summary dict.

        Args:
            transcript_text: Full transcript as plain text (segments joined by space).
            duration_secs:   Total video duration — used in critique for timestamp checks.
            video_title:     Optional title for context injection.
        """
        duration_ms = int(duration_secs * 1000)

        context_prefix = (
            f'Video title: "{video_title}"\n' if video_title else ""
        )
        full_transcript = context_prefix + transcript_text

        t0 = time.perf_counter()

        # ── Call 1: Draft ─────────────────────────────────────────────────
        draft_response = self.client.messages.create(
            model      = self.model,
            max_tokens = 4096,
            system     = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},   # cache system prompt
                }
            ],
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<transcript>\n{full_transcript}\n</transcript>",
                            "cache_control": {"type": "ephemeral"},  # cache transcript
                        },
                        {
                            "type": "text",
                            "text": DRAFT_TASK_PROMPT.format(transcript="[see above]"),
                        },
                    ],
                }
            ],
        )

        draft_text = draft_response.content[0].text
        draft_dict = _extract_json(draft_text)
        draft_json = json.dumps(draft_dict, indent=2)

        call1_usage = draft_response.usage
        logger.info(
            "summary_draft_generated",
            input_tokens         = call1_usage.input_tokens,
            output_tokens        = call1_usage.output_tokens,
            cache_creation_tokens= getattr(call1_usage, "cache_creation_input_tokens", 0),
            cache_read_tokens    = getattr(call1_usage, "cache_read_input_tokens", 0),
        )

        # ── Call 2: Self-critique + Refine ────────────────────────────────
        critique_response = self.client.messages.create(
            model      = self.model,
            max_tokens = 6144,   # critique text + full refined JSON
            system     = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},   # cache hit
                }
            ],
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<transcript>\n{full_transcript}\n</transcript>",
                            "cache_control": {"type": "ephemeral"},  # cache hit — ~10x cheaper
                        },
                        {
                            "type": "text",
                            "text": CRITIQUE_PROMPT.format(
                                draft        = draft_json,
                                duration_secs= duration_secs,
                                duration_ms  = duration_ms,
                            ),
                        },
                    ],
                }
            ],
        )

        critique_text = critique_response.content[0].text
        critique_notes = _extract_critique(critique_text)
        refined_dict   = _extract_json(critique_text)

        call2_usage = critique_response.usage
        elapsed = time.perf_counter() - t0

        logger.info(
            "summary_refined_generated",
            input_tokens          = call2_usage.input_tokens,
            output_tokens         = call2_usage.output_tokens,
            cache_creation_tokens = getattr(call2_usage, "cache_creation_input_tokens", 0),
            cache_read_tokens     = getattr(call2_usage, "cache_read_input_tokens", 0),
            elapsed_secs          = round(elapsed, 2),
            critique_severity     = _extract_severity(critique_notes),
        )

        # Attach metadata to the result
        refined_dict["_meta"] = {
            "model":            self.model,
            "critique_notes":   critique_notes,
            "critique_severity":_extract_severity(critique_notes),
            "draft_snapshot":   draft_dict,    # keep draft for audit/comparison
            "total_input_tokens": (
                call1_usage.input_tokens + call2_usage.input_tokens
            ),
            "total_output_tokens": (
                call1_usage.output_tokens + call2_usage.output_tokens
            ),
        }

        return refined_dict


def _extract_severity(critique_text: str) -> str:
    m = re.search(r"OVERALL_SEVERITY:\s*(low|medium|high)", critique_text, re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"
