"""
Quick edge case test — validates Fireworks AI llama-3.1-8b-instruct
against known tricky comments before we build the full pipeline.
"""

import json
from openai import OpenAI

client = OpenAI(
    api_key="fw_NawHB6b9Lanvkgc6axSdJL",
    base_url="https://api.fireworks.ai/inference/v1"
)

SYSTEM_PROMPT = """You are a production-grade educational comment classifier.
Classify YouTube comments on educational videos with precision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic: Science-based guide to protein intake for muscle building.
Examines the 1g/lb rule and gives evidence-based recommendations (0.36–2.2g/kg).

TOPICS THE VIDEO EXPLICITLY COVERED (use for answered_by_video):
  ✓ Protein targets by training phase: bulking 1.6–2.2g/kg, cutting toward upper end
  ✓ Lean body mass calculation for adjusting protein (Eric Helms research)
  ✓ 3g leucine threshold to trigger mTOR / muscle protein synthesis
  ✓ Post-workout window is 4–6 hours, NOT 30 minutes
  ✓ Protein spread across 3–5 meals keeps synthesis elevated
  ✓ High protein is safe — ISSN: no health risks in healthy individuals
  ✓ Research supports up to 4.4g/kg in healthy people
  ✓ 1g/lb rule is contextual, not universal
  ✓ Animal protein hits leucine threshold with less food than plant protein
  ✓ DIAAS amino acid scoring system
  ✗ Protein on rest days — NOT covered
  ✗ Protein for older adults / elderly — NOT covered
  ✗ Protein for weight loss without training — NOT covered
  ✗ Specific meal plans or food lists — NOT covered
  ✗ Supplements beyond whey/casein mention — NOT covered

KEY CLAIMS (cross-check for misconception detection):
  - 1g/lb is contextual, not a universal rule
  - 30-minute anabolic window is a myth; window is 4–6 hours
  - High protein does NOT damage kidneys in healthy individuals
  - Plant protein requires more food to hit leucine threshold vs animal

KNOWN CONTROVERSY POINTS:
  - Many viewers believe 1g/lb is mandatory regardless of phase
  - Plant protein community may push back on leucine comparison
  - 30-minute post-workout window is widely believed in gyms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT LABELS  —  assign ALL that apply
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
question      → asks something OR expresses uncertainty (with or without "?")
                Triggers: "wondering if", "not sure about", "does this apply", "what about",
                "can someone explain", "I don't know if", "should I", "how do I"

praise        → genuine positive reaction to content quality, explanation, or creator

criticism     → disagrees with a claim, method, or conclusion in the video
                CO-LABEL RULE: misconception stated dismissively or assertively
                ALSO gets "criticism". Example: "high protein wrecks kidneys tho" = misconception + criticism

confusion     → does not understand something from the video
                CO-LABEL RULE: sarcastic "I totally understand" = confusion + criticism

misconception → viewer states something as fact that CONTRADICTS the video's key claims
                Must cross-check against KEY CLAIMS above. If not in key claims, do NOT label misconception.

request       → asks for future content, more detail, or a follow-up video

spam          → purely promotional, bot-like, self-promotional, @mention only, irrelevant links
                NOTE: short positive reactions ("facts", "true", "same") are NOT spam → label praise or off_topic

off_topic     → completely unrelated to video content (creator appearance, unrelated life story)
                Timestamp references ("at 3:45 he says...") are NOT off_topic → classify by what they say about it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENTIMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
positive / neutral / negative

EMOJI MEANINGS for this audience (young adults, Gen Z, casual fitness):
  💀 💀💀   = shocked/hilarious (extremely POSITIVE, not death)
  😭        = overwhelmed/funny (POSITIVE, not sad)
  🔥 ❤️ 👏 🙌 🤯 = POSITIVE
  😂 🤣     = laughing = POSITIVE
  👎 😤 🙄 😒 🤬 = NEGATIVE
  🤔 😐     = NEUTRAL
  RULE: when emojis conflict with text, EMOJIS WIN for this audience

LABEL-SPECIFIC SENTIMENT DEFAULTS (override only with strong opposing signal):
  request   → positive or neutral (almost never negative)
  praise    → positive
  spam      → neutral
  misconception alone (stated calmly) → neutral
  misconception + criticism → negative

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SARCASM — classify by TRUE meaning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sarcasm signals: 🙄 😒 😤, "oh sure", "yeah right", "totally", "definitely",
                 "clearly", "obviously" used with no genuine enthusiasm

Examples:
  "Oh yeah 1g/lb works perfectly for everyone 🙄"   → criticism (NOT praise)
  "Wow super clear explanation, not confused at all" → confusion + criticism (NOT praise)
  "Sure, just eat more protein, problem solved 😒"   → criticism (NOT praise)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPLY CONTEXT RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When parent_comment is provided, evaluate the reply IN THAT CONTEXT.
"Exactly" agreeing with a misconception parent → also misconception
"Wrong" disagreeing with a misconception parent → criticism

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT: "high protein wrecks your kidneys tho"
OUTPUT: intent_labels=["misconception","criticism"], sentiment="negative"
WHY: contradicts key claim + dismissive tone triggers co-label criticism

INPUT: "great video but what about protein on rest days?"
OUTPUT: intent_labels=["praise","question"], sentiment="positive", answered_by_video=false
WHY: rest days NOT in covered topics list

INPUT: "💀💀💀"
OUTPUT: intent_labels=["praise"], sentiment="positive"
WHY: 💀 = Gen Z positive reaction

INPUT: "Wow super clear, not confused at all 😒"
OUTPUT: intent_labels=["confusion","criticism"], sentiment="negative"
WHY: sarcasm + 😒 = true meaning is confusion + criticism

INPUT: "please make a video on plant-based protein"
OUTPUT: intent_labels=["request"], sentiment="positive"
WHY: request sentiment default is positive

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
answered_by_video  (ONLY for comments with "question" label)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
true  → topic appears in TOPICS THE VIDEO EXPLICITLY COVERED list above
false → topic does NOT appear in that list
When uncertain, default to false.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — return ONLY a valid JSON array, no markdown, no explanation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {
    "comment_id": "...",
    "intent_labels": ["question", "confusion"],
    "sentiment": "neutral",
    "answered_by_video": true,
    "confidence": 0.91
  }
]

HARD RULES:
- Always assign at least 1 intent label, never an empty array
- Emoji-only: classify by emoji meaning, NEVER spam
- "facts", "true", "same", "real" alone → praise or off_topic, NOT spam
- Multi-label freely — realistic comments have 2–3 labels often
- answered_by_video only included when "question" is in intent_labels
- confidence < 0.65 means genuinely ambiguous — still classify, just report low confidence
"""

EDGE_CASE_COMMENTS = [
    {"comment_id": "ec_01", "text": "💀💀💀",                                                                     "expected": "praise / positive"},
    {"comment_id": "ec_02", "text": "this destroyed me 😭",                                                       "expected": "praise / positive"},
    {"comment_id": "ec_03", "text": "Oh yeah 1g per pound totally works for everyone 🙄",                         "expected": "criticism / negative (sarcasm)"},
    {"comment_id": "ec_04", "text": "wondering if this applies when I'm cutting",                                  "expected": "question / neutral / answered_by_video=true"},
    {"comment_id": "ec_05", "text": "high protein wrecks your kidneys tho",                                        "expected": "misconception+criticism / negative"},
    {"comment_id": "ec_06", "text": "great video but what about protein on rest days?",                            "expected": "praise+question / positive / answered_by_video=false"},
    {"comment_id": "ec_07", "text": "I'm still confused about the leucine threshold",                             "expected": "confusion / neutral"},
    {"comment_id": "ec_08", "text": "please make a video on plant-based protein sources",                         "expected": "request / positive"},
    {"comment_id": "ec_09", "text": "subscribe to my channel 👇",                                                 "expected": "spam"},
    {"comment_id": "ec_10", "text": "you only need to eat protein right after workout, 30 min window is real",    "expected": "misconception / negative"},
    {"comment_id": "ec_11", "text": "🔥🔥",                                                                       "expected": "praise / positive"},
    {"comment_id": "ec_12", "text": "Wow super clear explanation, not confused at all 😒",                        "expected": "confusion+criticism / negative (sarcasm)"},
    {"comment_id": "ec_13", "text": "does the 1g per pound rule apply if I'm overweight",                         "expected": "question / neutral / answered_by_video=true (lean body mass covered)"},
    {"comment_id": "ec_14", "text": "been doing 4 meals a day for years, finally understand why it works 🙌",     "expected": "praise / positive"},
    {"comment_id": "ec_15", "text": "plant protein is just as good as animal protein for leucine",                "expected": "misconception / neutral"},
]

user_message = "Classify these comments:\n\n" + "\n".join(
    f'{i+1}. [comment_id: {c["comment_id"]}] {c["text"]}'
    for i, c in enumerate(EDGE_CASE_COMMENTS)
)

print("Sending 15 edge case comments to Fireworks llama-3.1-8b-instruct...\n")

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ],
    temperature=0.1,
    max_tokens=3500,
)

raw = response.choices[0].message.content.strip()

# parse JSON
start = raw.find("[")
end   = raw.rfind("]") + 1
try:
    results = json.loads(raw[start:end])
except json.JSONDecodeError as e:
    print("RAW OUTPUT:\n", raw)
    raise e

print(f"{'ID':<8} {'EXPECTED':<52} {'GOT LABELS':<40} {'SENTIMENT':<10} {'CONF'}")
print("-" * 130)
for expected, got in zip(EDGE_CASE_COMMENTS, results):
    labels   = "+".join(got.get("intent_labels", []))
    sentiment = got.get("sentiment", "?")
    conf      = got.get("confidence", 0)
    avid      = f" avid={got['answered_by_video']}" if "answered_by_video" in got else ""
    print(f"{expected['comment_id']:<8} {expected['expected']:<52} {labels+avid:<40} {sentiment:<10} {conf:.2f}")

print(f"\nTokens used — prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}")
