"""Sarvam chat completions for grounded Indic / English answers."""

from __future__ import annotations

from . import config

SYSTEM_DEFAULT = """You are an educational assistant about Ayurveda (traditional wellness knowledge from South Asia).
Rules:
- Use ONLY the evidence in CONTEXT. If it is missing or insufficient, say you do not have enough evidence.
- Cite each claim with the bracket tags from CONTEXT, e.g. [pdf:...] or [dataset row ...].
- Prefer clear bullet lists for remedies or lifestyle points when appropriate.
- This is NOT medical diagnosis or prescription; add a brief disclaimer.
- Respond in the same language as the user when you can (including Indic scripts)."""


def generate_answer(system_prompt: str, context_block: str, user_message: str) -> str:
    if not config.SARVAM_API_KEY:
        return (
            "**Sarvam API key not set** (`SARVAM_API_KEY`). Showing retrieved context only:\n\n"
            + context_block[:6000]
        )
    try:
        from sarvamai import SarvamAI
    except ImportError:
        return (
            "**Install `sarvamai`** for LLM responses. Context:\n\n" + context_block[:6000]
        )
    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    user_block = f"CONTEXT:\n{context_block}\n\nUSER MESSAGE:\n{user_message}"
    kwargs = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_block}], "temperature": 0.25}
    if config.SARVAM_MODEL:
        kwargs["model"] = config.SARVAM_MODEL
    try:
        resp = client.chat.completions(**kwargs)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return (
            f"**Sarvam request failed** ({type(e).__name__}: {e}).\n\n"
            f"Context excerpt:\n\n{context_block[:4000]}"
        )
