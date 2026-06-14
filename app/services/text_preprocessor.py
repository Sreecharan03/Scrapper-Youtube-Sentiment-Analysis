"""
app/services/text_preprocessor.py
===================================
Pre-processes comment text before sending to the LLM classifier.

Rules applied in order:
  1. Skip pure-number comments  → returns None (comment is skipped)
  2. Replace emojis inline      → 💀 becomes [skull]
  3. Clean punctuation          → collapses repeated symbols, removes junk

Emoji dictionary is loaded once at import time from emoji_dict.json
(scraped from https://unicode.org/emoji/charts/full-emoji-list.html).
"""

import json
import re
from pathlib import Path

# ── Load emoji dictionary ─────────────────────────────────────────────────────

_DICT_PATH = Path(__file__).parent / "emoji_dict.json"

with open(_DICT_PATH, encoding="utf-8") as _f:
    _EMOJI_DICT: dict[str, str] = json.load(_f)

# Sort longest-first so multi-codepoint sequences (👨‍👩‍👧) match before sub-sequences
_EMOJI_SORTED: list[str] = sorted(_EMOJI_DICT.keys(), key=len, reverse=True)


# ── Rule 1 — Pure-number filter ───────────────────────────────────────────────

_PURE_NUMBER_RE = re.compile(r'^[\d\s,.\-+]+$')

def _is_pure_numbers(text: str) -> bool:
    """
    Returns True if the comment is nothing but digits, spaces, and
    number-formatting characters (commas, dots, dashes, plus).
    Examples: "1000", "1,000", "42 100", "3.14" → True
    "1st place", "100%", "1 protein" → False
    """
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_PURE_NUMBER_RE.match(stripped)) and any(c.isdigit() for c in stripped)


# ── Rule 2 — Emoji replacement ────────────────────────────────────────────────

def _replace_emojis(text: str) -> str:
    """
    Replace each emoji with [description] inline.
    Longest sequences matched first to handle ZWJ / skin-tone sequences.

    Example:
        "great video 💀💀"  →  "great video [skull] [skull]"
        "💀😭🔥"            →  "[skull] [loudly crying face] [fire]"
    """
    for emoji in _EMOJI_SORTED:
        if emoji in text:
            name = _EMOJI_DICT[emoji]
            # Wrap with spaces so adjacent emojis don't merge: 💀💀 → [skull] [skull]
            text = text.replace(emoji, f" [{name}] ")
    # Normalize extra spaces introduced by replacements
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# ── Rule 3 — Punctuation cleaning ────────────────────────────────────────────

# Patterns: (compiled_regex, replacement)
_PUNCT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\.{2,}'),   '.'),   # .... → .
    (re.compile(r',{2,}'),    ','),   # ,,,, → ,
    (re.compile(r'!{2,}'),    '!'),   # !!!! → !
    (re.compile(r'\?{2,}'),   '?'),   # ???? → ?
    (re.compile(r'-{3,}'),    '--'),  # ---- → --
    (re.compile(r'\*{2,}'),   ''),    # **** → (removed — markdown bold)
    (re.compile(r'~{2,}'),    ''),    # ~~~~ → (removed — strikethrough)
    (re.compile(r'#{2,}'),    ''),    # #### → (removed — heading noise)
    (re.compile(r'_{2,}'),    ''),    # ____ → (removed — underline noise)
    (re.compile(r'[\x00-\x08\x0b-\x1f\x7f]'), ' '),  # control chars → space
    (re.compile(r'\s{2,}'),   ' '),   # multiple spaces → single space
]

def _clean_punctuation(text: str) -> str:
    """
    Normalise punctuation without stripping Unicode letters.
    Only collapses repeated symbols and removes formatting noise.
    Does NOT strip accented letters, CJK, Arabic, etc.
    """
    for pattern, replacement in _PUNCT_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str | None:
    """
    Full pre-processing pipeline for a single comment.

    Returns:
        Cleaned string ready for the LLM, or
        None if the comment should be skipped entirely.

    Skipped when:
        - Empty / whitespace only
        - Pure numbers (e.g. "1000", "42 100")
        - Becomes empty after cleaning
    """
    if not text or not text.strip():
        return None

    if _is_pure_numbers(text):
        return None

    text = _replace_emojis(text)
    text = _clean_punctuation(text)

    return text if text.strip() else None


def preprocess_with_parent(comment: dict) -> dict | None:
    """
    Pre-process a comment dict {comment_id, text, parent_text?}.

    Returns updated dict with cleaned text/parent_text, or
    None if the comment should be skipped.
    """
    cleaned = preprocess(comment.get("text", ""))
    if cleaned is None:
        return None

    result = {**comment, "text": cleaned}

    if "parent_text" in comment:
        cleaned_parent = preprocess(comment["parent_text"])
        if cleaned_parent:
            result["parent_text"] = cleaned_parent
        else:
            result.pop("parent_text", None)

    return result
