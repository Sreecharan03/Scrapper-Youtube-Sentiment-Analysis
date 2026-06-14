"""
app/services/classifier.py
============================
Comment classification service using Fireworks AI (DeepSeek V4 Flash).

DESIGN:
  - System prompt is built dynamically from the video's LLM summary JSON so
    every classification is grounded in what the video actually said.
  - Comments batched 50 per API call to minimize request overhead.
  - Replies include parent TLC text (first 200 chars) for context.
  - Non-English detection via Unicode letter range check — emoji-only comments
    are NOT skipped (emojis carry sentiment signal for this audience).
  - asyncio.Semaphore(10) limits concurrent API calls.
  - Exponential backoff (1s, 2s, 4s) on transient API errors.
"""

import asyncio
import json
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.services.relevance_filter import RelevanceFilter
from app.services.text_preprocessor import preprocess_with_parent

logger = get_logger(__name__)

BATCH_SIZE             = 25   # 8b models struggle with 50-item JSON — 25 is reliable
MAX_CONCURRENCY        = 2    # Groq: 30 RPM free tier — keep conservative
MAX_RETRIES            = 4
CLASSIFICATION_VERSION = "v1"
GROQ_BASE_URL          = "https://api.groq.com/openai/v1"


# ── Language detection ────────────────────────────────────────────────────────

def _is_non_english(text: str) -> bool:
    """
    Returns True if the text is likely non-English and should be skipped.

    Strategy: count Unicode letter characters. If >30% fall outside the
    Latin/Latin-Extended range (ord > 0x024F), classify as non-English.
    Emoji-only comments have no letters → never marked non-English.
    Latin-script languages (Spanish, French, Portuguese) pass through.
    """
    if not text:
        return False
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if len(letters) < 3:
        return False
    non_latin = sum(1 for c in letters if ord(c) > 0x024F)
    return (non_latin / len(letters)) > 0.30


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_system_prompt(summary: dict) -> str:
    """Build a context-aware classification prompt from the video's summary JSON."""

    overview   = summary.get("overview", "Educational video.")
    topics     = summary.get("key_topics", [])
    claims     = summary.get("key_claims", [])
    triggers   = summary.get("controversy_triggers", [])

    covered_lines  = "\n".join(f"  ✓ {t['label']}: {t.get('description', '')}" for t in topics)
    claims_lines   = "\n".join(f"  - {c}" for c in claims)
    triggers_lines = "\n".join(f"  - {t}" for t in triggers)

    return f"""You are a production-grade educational comment classifier.
Classify YouTube comments on educational videos with precision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic: {overview}

TOPICS THE VIDEO EXPLICITLY COVERED (use for answered_by_video):
{covered_lines}
  ✗ Anything not listed above → answered_by_video: false

KEY CLAIMS (cross-check for misconception detection):
{claims_lines}

KNOWN CONTROVERSY POINTS:
{triggers_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT LABELS  —  assign ALL that apply
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
question      → asks something OR expresses uncertainty (with or without "?")
                Triggers: "wondering if", "not sure about", "does this apply",
                "what about", "can someone explain", "should I", "how do I"

praise        → genuine positive reaction to content quality, explanation, or creator

criticism     → disagrees with a claim, method, or conclusion in the video
                CO-LABEL RULE: misconception stated dismissively or assertively
                ALSO gets "criticism"

confusion     → does not understand something from the video
                CO-LABEL RULE: sarcastic "I totally understand" = confusion + criticism

misconception → viewer states something as fact that CONTRADICTS the video's key claims
                Must cross-check against KEY CLAIMS. Not in list → do NOT label misconception.

request       → asks for future content, more detail, or a follow-up video

spam          → purely promotional, bot-like, @mention only, irrelevant links
                NOTE: short reactions ("facts", "true", "same") are NOT spam → praise/off_topic

off_topic     → completely unrelated to video content

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENTIMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
positive / neutral / negative

EMOJI MEANINGS for this audience (young adults, Gen Z, casual):
  💀 💀💀   = shocked/hilarious → POSITIVE
  😭        = overwhelmed/funny → POSITIVE (NOT sad)
  🔥 ❤️ 👏 🙌 🤯 😂 🤣 = POSITIVE
  👎 😤 🙄 😒 🤬 = NEGATIVE
  🤔 😐     = NEUTRAL
  RULE: when emojis conflict with text, EMOJIS WIN

LABEL-SPECIFIC SENTIMENT DEFAULTS:
  request        → positive or neutral (almost never negative)
  praise         → positive
  spam           → neutral
  misconception alone (stated calmly)  → neutral
  misconception + criticism            → negative

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SARCASM — classify by TRUE meaning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signals: 🙄 😒 😤, "oh sure", "yeah right", "totally", "clearly" used ironically
"Oh yeah totally works for everyone 🙄"    → criticism NOT praise
"Wow super clear, not confused at all 😒"  → confusion + criticism NOT praise

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPLY CONTEXT RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When parent_comment is provided, evaluate the reply IN THAT CONTEXT.
"Exactly" agreeing with a misconception → also misconception.
"Wrong" disagreeing with a misconception → criticism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT: "high protein wrecks your kidneys tho"
OUTPUT: intent_labels=["misconception","criticism"], sentiment="negative"

INPUT: "great video but what about protein on rest days?"
OUTPUT: intent_labels=["praise","question"], sentiment="positive", answered_by_video=false

INPUT: "💀💀💀"
OUTPUT: intent_labels=["praise"], sentiment="positive"

INPUT: "Wow super clear, not confused at all 😒"
OUTPUT: intent_labels=["confusion","criticism"], sentiment="negative"

INPUT: "please make a video on plant-based protein"
OUTPUT: intent_labels=["request"], sentiment="positive"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
answered_by_video  (ONLY for comments with "question" label)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
true  → topic is in TOPICS THE VIDEO EXPLICITLY COVERED list
false → not in that list. When uncertain, default to false.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — return ONLY a valid JSON object, no markdown, no explanation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "classifications": [
    {{
      "comment_id": "...",
      "intent_labels": ["question", "confusion"],
      "sentiment": "neutral",
      "answered_by_video": true,
      "confidence": 0.91
    }}
  ]
}}

HARD RULES:
- Always assign at least 1 intent label, never an empty array
- Emoji-only: classify by emoji meaning, NEVER spam
- "facts", "true", "same" alone → praise or off_topic, NOT spam
- Multi-label freely — realistic comments often have 2–3 labels
- answered_by_video only included when "question" is in intent_labels
- confidence < 0.65 = ambiguous — still classify, report low confidence
"""


# ── Classifier ────────────────────────────────────────────────────────────────

class CommentClassifier:
    """
    Classifies YouTube comments in batches using Fireworks AI.

    Usage:
        classifier = CommentClassifier(api_key=..., model=...)
        results, skipped, failed = await classifier.classify_all(comments, summary)
    """

    def __init__(self, api_key: str, model: str, base_url: str = GROQ_BASE_URL) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model

    async def _classify_batch(
        self,
        batch: list[dict],
        system_prompt: str,
        semaphore: asyncio.Semaphore,
    ) -> list[dict]:
        """
        Classify one batch (up to BATCH_SIZE comments) with retry + backoff.
        Returns empty list if all retries exhausted.
        """
        user_message = "Classify these comments:\n\n" + "\n".join(
            "{n}. [comment_id: {cid}]{parent} {text}".format(
                n=i + 1,
                cid=c["comment_id"],
                parent=f" [parent: {c['parent_text'][:150]}]" if c.get("parent_text") else "",
                text=c["text"],
            )
            for i, c in enumerate(batch)
        )

        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=0.1,
                        max_tokens=3500,
                        response_format={"type": "json_object"},
                    )

                raw  = response.choices[0].message.content.strip()
                data = json.loads(raw)
                # Support both {"classifications": [...]} wrapper and bare array fallback
                if isinstance(data, dict):
                    items = data.get("classifications") or data.get("results") or []
                    if not items:
                        # try first list value in the dict
                        for v in data.values():
                            if isinstance(v, list):
                                items = v
                                break
                    return items
                if isinstance(data, list):
                    return data
                raise ValueError(f"Unexpected JSON shape: {type(data)}")

            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    logger.error(
                        "classification_batch_failed_permanently",
                        attempt=attempt + 1,
                        error=str(exc),
                        batch_size=len(batch),
                    )
                    return []

                # 429 rate limit — back off much longer
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                backoff = 60 if is_rate_limit else (2 ** attempt)
                logger.warning(
                    "classification_batch_retrying",
                    attempt=attempt + 1,
                    backoff=backoff,
                    rate_limited=is_rate_limit,
                    error=str(exc),
                )
                await asyncio.sleep(backoff)

        return []

    async def classify_all(
        self,
        comments: list[dict],
        summary: dict,
    ) -> tuple[list[dict], int, int]:
        """
        Classify all comments for a video.

        Args:
            comments: list of {comment_id, text, parent_text?}
            summary:  video summary dict (from summaries collection)

        Returns:
            (results, skipped_count, failed_count)
            Each result: {comment_id, classification_status, intent_labels?,
                          sentiment?, answered_by_video?, classification_confidence?}
        """
        system_prompt = _build_system_prompt(summary)

        to_classify: list[dict] = []
        skipped_ids: list[str]  = []

        for c in comments:
            # Skip non-English
            if _is_non_english(c.get("text", "")):
                skipped_ids.append(c["comment_id"])
                continue

            # Apply pre-processing: emoji replacement, number filter, punctuation clean
            processed = preprocess_with_parent(c)
            if processed is None:
                # Pure numbers or empty after cleaning — skip
                skipped_ids.append(c["comment_id"])
                continue

            to_classify.append(processed)

        if skipped_ids:
            logger.info("comments_skipped", count=len(skipped_ids))

        # ── Relevance filter: replies only ────────────────────────────────
        # TLCs always pass through. Only replies are checked for semantic
        # relevance against the parent comment + video topics.
        tlcs    = [c for c in to_classify if not c.get("is_reply", False)]
        replies = [c for c in to_classify if c.get("is_reply", False)]

        if replies:
            rf = RelevanceFilter(summary)
            relevant_replies, irrelevant_ids = rf.filter_replies(replies)
            skipped_ids.extend(irrelevant_ids)
            to_classify = tlcs + relevant_replies
            logger.info(
                "relevance_filter_applied",
                tlcs=len(tlcs),
                replies_total=len(replies),
                replies_kept=len(relevant_replies),
                replies_dropped=len(irrelevant_ids),
            )
        else:
            to_classify = tlcs

        batches = [
            to_classify[i : i + BATCH_SIZE]
            for i in range(0, len(to_classify), BATCH_SIZE)
        ]

        semaphore    = asyncio.Semaphore(MAX_CONCURRENCY)
        batch_tasks  = [self._classify_batch(b, system_prompt, semaphore) for b in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        now          = datetime.now(timezone.utc)
        all_results: list[dict] = []
        failed_count = 0

        for batch, raw_results in zip(batches, batch_results):
            if not raw_results:
                failed_count += len(batch)
                for c in batch:
                    all_results.append({"comment_id": c["comment_id"], "classification_status": "failed"})
                continue

            result_map = {r["comment_id"]: r for r in raw_results if "comment_id" in r}

            for c in batch:
                cid = c["comment_id"]
                r   = result_map.get(cid)

                if r is None:
                    failed_count += 1
                    all_results.append({"comment_id": cid, "classification_status": "failed"})
                    continue

                entry: dict = {
                    "comment_id":               cid,
                    "intent_labels":            r.get("intent_labels") or ["off_topic"],
                    "sentiment":                r.get("sentiment", "neutral"),
                    "classification_confidence": float(r.get("confidence", 0.0)),
                    "classification_status":    "done",
                    "classified_at":            now,
                    "classification_version":   CLASSIFICATION_VERSION,
                }
                # Only write answered_by_video when the model returned it
                if "question" in entry["intent_labels"] and "answered_by_video" in r:
                    entry["answered_by_video"] = r["answered_by_video"]

                all_results.append(entry)

        # Append skipped entries
        for cid in skipped_ids:
            all_results.append({
                "comment_id":             cid,
                "classification_status":  "skipped",
                "classified_at":          now,
                "classification_version": CLASSIFICATION_VERSION,
            })

        logger.info(
            "classification_all_done",
            total=len(comments),
            classified=len(to_classify) - failed_count,
            skipped=len(skipped_ids),
            failed=failed_count,
        )

        return all_results, len(skipped_ids), failed_count
