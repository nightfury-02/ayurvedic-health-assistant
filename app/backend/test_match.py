"""Lightweight rule-based symptom matcher used by the Streamlit demo.

This is intentionally simple (no ML) so the app can run with zero external
dependencies. It picks the most relevant entry from `DATA` based on how many
keyword tokens appear in the user input as whole words.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List

DATA: List[Dict[str, Any]] = [
    {
        "disease": "Cough",
        "keywords": ["cough", "throat", "congestion", "khansi", "phlegm", "cold"],
        "dosha": ["Kapha", "Vata"],
        "advice": "Drink warm fluids, avoid cold foods, try ginger-tulsi tea.",
    },
    {
        "disease": "Diabetes",
        "keywords": ["fatigue", "urination", "sugar", "thirst", "diabetes"],
        "dosha": ["Kapha", "Pitta"],
        "advice": "Avoid sugar, follow a low-GI diet, exercise regularly.",
    },
    {
        "disease": "Hypertension",
        "keywords": ["bp", "pressure", "stress", "hypertension", "headache"],
        "dosha": ["Pitta", "Vata"],
        "advice": "Reduce salt, practice meditation and pranayama.",
    },
]

DEFAULT_RESULT: Dict[str, Any] = {
    "disease": "General imbalance",
    "dosha": ["Vata"],
    "advice": "Improve sleep, reduce stress, and eat a balanced, warm diet.",
    "matched_keywords": [],
    "confidence": 0.0,
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _score(tokens_set: set, keywords: List[str]) -> tuple[int, List[str]]:
    matched = [k for k in keywords if k.lower() in tokens_set]
    return len(matched), matched


def predict(text: str) -> Dict[str, Any]:
    """Return the best matching disease entry for the given symptom text.

    Uses whole-word token matching to avoid false positives like 'sugar' inside
    'sugarcane'. Returns a copy so callers cannot mutate the underlying DATA.
    """
    tokens = set(_tokenize(text))
    if not tokens:
        return deepcopy(DEFAULT_RESULT)

    best_entry: Dict[str, Any] | None = None
    best_score = 0
    best_matches: List[str] = []

    for row in DATA:
        score, matches = _score(tokens, row["keywords"])
        if score > best_score:
            best_score = score
            best_entry = row
            best_matches = matches

    if best_entry is None or best_score == 0:
        return deepcopy(DEFAULT_RESULT)

    result = deepcopy(best_entry)
    result["matched_keywords"] = best_matches
    result["confidence"] = round(best_score / max(len(best_entry["keywords"]), 1), 2)
    return result
