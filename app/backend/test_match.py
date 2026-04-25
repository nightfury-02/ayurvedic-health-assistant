"""Legacy keyword stub — superseded by `rag.run_chat` / `rag.legacy_symptom_card`."""

from __future__ import annotations

from backend.rag import legacy_symptom_card


def predict(text: str) -> dict:
    return legacy_symptom_card(text)
