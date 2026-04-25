"""Lightweight multilingual helpers for the RAG pipeline.

The Ayurveda knowledge base is indexed in English. To support regional Indian
languages without depending on an external translation API, we:

1. Detect the user's language from the **Unicode script** of the query
   (instant, deterministic, zero-dependency).
2. Build chat-completion messages that ask the existing Databricks LLM
   (e.g. ``databricks-meta-llama-3-3-70b-instruct``) to translate
   query → English before retrieval, and answer → user-language after
   generation.

The actual LLM invocation lives in the calling notebook; this module only
returns prompt payloads so it stays free of Databricks SDK imports and is
easily testable on its own.

Supported scripts (and the languages they cover):

* Devanagari   → Hindi (default), Marathi
* Bengali      → Bengali, Assamese
* Gurmukhi     → Punjabi
* Gujarati     → Gujarati
* Odia         → Odia
* Tamil        → Tamil
* Telugu       → Telugu
* Kannada      → Kannada
* Malayalam    → Malayalam
* Perso-Arabic → Urdu
* Latin        → English (no translation performed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# (lang_code, lang_name, codepoint_start, codepoint_end_inclusive)
_SCRIPT_RANGES: List[Tuple[str, str, int, int]] = [
    ("hi", "Hindi",     0x0900, 0x097F),  # Devanagari (also Marathi, Sanskrit)
    ("bn", "Bengali",   0x0980, 0x09FF),
    ("pa", "Punjabi",   0x0A00, 0x0A7F),  # Gurmukhi
    ("gu", "Gujarati",  0x0A80, 0x0AFF),
    ("or", "Odia",      0x0B00, 0x0B7F),
    ("ta", "Tamil",     0x0B80, 0x0BFF),
    ("te", "Telugu",    0x0C00, 0x0C7F),
    ("kn", "Kannada",   0x0C80, 0x0CFF),
    ("ml", "Malayalam", 0x0D00, 0x0D7F),
    ("ur", "Urdu",      0x0600, 0x06FF),  # Perso-Arabic
]

# Public lookup: code → human-readable name (for prompts and API responses).
SUPPORTED_LANGUAGES: Dict[str, str] = {code: name for code, name, _, _ in _SCRIPT_RANGES}
SUPPORTED_LANGUAGES["en"] = "English"


@dataclass(frozen=True)
class DetectedLanguage:
    code: str
    name: str
    is_english: bool


def detect_language(text: str) -> DetectedLanguage:
    """Detect the dominant script in ``text`` and map it to a language.

    Returns ``DetectedLanguage("en", "English", True)`` for empty input or
    text that contains no codepoints from any supported non-Latin script.
    """
    if not text:
        return DetectedLanguage("en", "English", True)

    counts: Dict[Tuple[str, str], int] = {}
    for ch in text:
        cp = ord(ch)
        for code, name, start, end in _SCRIPT_RANGES:
            if start <= cp <= end:
                counts[(code, name)] = counts.get((code, name), 0) + 1
                break

    if not counts:
        return DetectedLanguage("en", "English", True)

    (code, name), _ = max(counts.items(), key=lambda kv: kv[1])
    return DetectedLanguage(code=code, name=name, is_english=False)


def resolve_language(code_or_name: str) -> DetectedLanguage:
    """Resolve a user-supplied language code or name to a :class:`DetectedLanguage`.

    Accepts ISO-like codes (``"te"``) or human names (``"Telugu"``,
    case-insensitive). Falls back to English on unknown input.
    """
    if not code_or_name:
        return DetectedLanguage("en", "English", True)
    key = code_or_name.strip().lower()
    for code, name in SUPPORTED_LANGUAGES.items():
        if key == code.lower() or key == name.lower():
            return DetectedLanguage(code=code, name=name, is_english=(code == "en"))
    return DetectedLanguage("en", "English", True)


def build_translate_messages(
    text: str,
    source_language_name: str,
    target_language_name: str,
) -> List[Dict[str, str]]:
    """Build OpenAI-style chat messages for a translation request.

    The system prompt instructs the LLM to:

    * preserve named entities, Sanskrit/Latin scientific terms, and numbers,
    * preserve bracketed citations like ``[source_file, page]`` verbatim, and
    * return only the translation (no preamble, no quotes).
    """
    system = (
        "You are a precise translator. Translate the user's message from "
        f"{source_language_name} to {target_language_name}. "
        "Preserve named entities, numbers, Sanskrit and Latin scientific "
        "terms (for example: Vata, Pitta, Kapha, Ashwagandha, "
        "Withania somnifera, Triphala), and any bracketed citations such "
        "as [source_file, page] EXACTLY as written. Return ONLY the "
        "translation, with no preamble, no quotation marks, and no "
        "explanations."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
