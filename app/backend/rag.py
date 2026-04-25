"""Hybrid RAG: Vector Search chunks + SQL curated rows + Sarvam."""

from __future__ import annotations

import json
from typing import Any

from . import config, local_retrieval, sarvam_llm
from .curated_sql import search_curated_sql
from .vector_retrieval import search_chunks


def _row_get(row: dict[str, Any], *keys: str) -> str:
    """Support both Unity Catalog / Delta column names and raw Kaggle CSV headers."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _build_context(chunk_hits: list[dict], curated_hits: list[dict]) -> str:
    parts: list[str] = []
    for h in chunk_hits:
        text = str(h.get("chunk_text") or h.get("text") or "")[:1800]
        pk = h.get("chunk_pk") or h.get("id") or "?"
        sf = h.get("source_file", "?")
        pg = h.get("page_number", "?")
        ci = h.get("chunk_index", "?")
        parts.append(f"[pdf:{sf} page={pg} chunk_index={ci} id={pk}]\n{text}")
    for idx, row in enumerate(curated_hits, 1):
        disease = row.get("Disease") or row.get("disease") or "?"
        blob = json.dumps(row, ensure_ascii=False)[:2200]
        parts.append(f"[dataset row {idx} Disease={disease}]\n{blob}")
    if not parts:
        return "(No matching evidence in the knowledge base.)"
    return "\n\n---\n\n".join(parts)


def run_chat(user_message: str) -> dict[str, Any]:
    msg = (user_message or "").strip()
    warnings: list[str] = []
    chunk_hits: list[dict] = []
    curated_hits: list[dict] = []

    if config.USE_LOCAL_FALLBACK:
        warnings.append("Local CSV mode (set Databricks env + USE_LOCAL_FALLBACK=0 for lakehouse RAG).")
        curated_hits = local_retrieval.search_curated_local(msg, config.CURATED_SQL_LIMIT)
    else:
        try:
            chunk_hits = search_chunks(msg)
            if config.VECTOR_SEARCH_ENDPOINT and config.VECTOR_SEARCH_INDEX and not chunk_hits:
                warnings.append(
                    "Vector Search returned 0 hits (check index sync, columns, or endpoint)."
                )
        except Exception as e:
            warnings.append(f"Vector Search skipped: {e}")
            chunk_hits = []
        try:
            curated_hits = search_curated_sql(msg)
        except Exception as e:
            warnings.append(f"SQL curated search failed: {e}")
            curated_hits = local_retrieval.search_curated_local(msg, config.CURATED_SQL_LIMIT)

    ctx = _build_context(chunk_hits, curated_hits)
    answer = sarvam_llm.generate_answer(sarvam_llm.SYSTEM_DEFAULT, ctx, msg)
    return {
        "answer": answer,
        "chunk_hits": chunk_hits,
        "curated_hits": curated_hits,
        "warnings": warnings,
        "context_preview": ctx[:2500],
    }


def legacy_symptom_card(user_message: str) -> dict[str, Any]:
    """Shape compatible with the old Streamlit 'disease / dosha / advice' demo."""
    r = run_chat(user_message)
    rows = r.get("curated_hits") or []
    if not rows:
        ans = (r.get("answer") or "Try a longer symptom description.").strip()
        return {
            "disease": "No close match",
            "dosha": ["—"],
            "advice": ans[:4000],
            "warnings": r.get("warnings"),
        }
    row = rows[0]
    dosha_raw = _row_get(row, "Doshas")
    if "," in dosha_raw:
        dosha = [x.strip() for x in dosha_raw.split(",") if x.strip()]
    elif "-" in dosha_raw and len(dosha_raw) < 48:
        dosha = [x.strip() for x in dosha_raw.split("-") if x.strip()]
    else:
        dosha = [dosha_raw or "Vata"]
    diet = _row_get(row, "Diet_and_Lifestyle_Recommendations", "Diet and Lifestyle Recommendations")
    pat = _row_get(row, "Patient_Recommendations", "Patient Recommendations")
    herbs = _row_get(row, "Ayurvedic_Herbs", "Ayurvedic Herbs")
    form = _row_get(row, "Formulation")
    advice_bits = [x for x in (diet, pat, herbs, form) if x]
    advice = " ".join(advice_bits)[:1200] if advice_bits else (r.get("answer") or "")[:1200]
    return {
        "disease": _row_get(row, "Disease") or "Unknown",
        "dosha": dosha[:4],
        "advice": advice,
        "llm_answer": r.get("answer"),
        "warnings": r.get("warnings"),
    }
