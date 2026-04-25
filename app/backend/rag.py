"""Hybrid RAG: PDF Vector Search + curated Vector Search + optional SQL + Sarvam."""

from __future__ import annotations

import json
from typing import Any

from . import config, local_retrieval, sarvam_llm
from .curated_sql import search_curated_sql
from .vector_retrieval import search_curated_vector, search_pdf_chunks


def _row_get(row: dict[str, Any], *keys: str) -> str:
    """Support both Unity Catalog / Delta column names and raw Kaggle CSV headers."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _format_curated_vector_hit(h: dict[str, Any], rank: int) -> str:
    """Build LLM context from curated index payload (omit repetitive embedding_text)."""
    payload = {k: v for k, v in h.items() if k != "embedding_text"}
    head = f"[ayurgenix_curated_vector rank={rank} id={h.get('curated_pk', '?')}]"
    return f"{head}\n{json.dumps(payload, ensure_ascii=False)[:2400]}"


def _build_context(
    pdf_hits: list[dict[str, Any]],
    curated_vec_hits: list[dict[str, Any]],
    curated_sql_rows: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for h in pdf_hits:
        text = str(h.get("chunk_text") or h.get("text") or "")[:1800]
        pk = h.get("chunk_pk") or h.get("id") or "?"
        sf = h.get("source_file", "?")
        pg = h.get("page_number", "?")
        ci = h.get("chunk_index", "?")
        parts.append(f"[pdf:{sf} page={pg} chunk_index={ci} id={pk}]\n{text}")
    for i, h in enumerate(curated_vec_hits, 1):
        parts.append(_format_curated_vector_hit(h, i))
    for idx, row in enumerate(curated_sql_rows, 1):
        disease = row.get("Disease") or row.get("disease") or "?"
        blob = json.dumps(row, ensure_ascii=False)[:2200]
        parts.append(f"[dataset_sql row {idx} Disease={disease}]\n{blob}")
    if not parts:
        return "(No matching evidence in the knowledge base.)"
    return "\n\n---\n\n".join(parts)


def run_chat(user_message: str) -> dict[str, Any]:
    msg = (user_message or "").strip()
    warnings: list[str] = []
    pdf_hits: list[dict] = []
    curated_vec_hits: list[dict] = []
    curated_sql_rows: list[dict] = []

    if config.USE_LOCAL_FALLBACK:
        warnings.append("Local CSV mode (set Databricks env + USE_LOCAL_FALLBACK=0 for lakehouse RAG).")
        curated_sql_rows = local_retrieval.search_curated_local(msg, config.CURATED_SQL_LIMIT)
    else:
        try:
            pdf_hits = search_pdf_chunks(msg)
            if config.VECTOR_SEARCH_ENDPOINT and config.VECTOR_SEARCH_INDEX_PDF and not pdf_hits:
                warnings.append(
                    "PDF Vector Search returned 0 hits (check PDF index sync, columns, or endpoint)."
                )
        except Exception as e:
            warnings.append(f"PDF Vector Search skipped: {e}")
            pdf_hits = []

        try:
            curated_vec_hits = search_curated_vector(msg)
            if (
                config.VECTOR_SEARCH_ENDPOINT
                and config.VECTOR_SEARCH_INDEX_CURATED
                and not curated_vec_hits
            ):
                warnings.append(
                    "Curated Vector Search returned 0 hits (run notebook 06, sync index, or check query)."
                )
        except Exception as e:
            warnings.append(f"Curated Vector Search skipped: {e}")
            curated_vec_hits = []

        if not config.SQL_WAREHOUSE_CONFIGURED and not config.VECTOR_SEARCH_INDEX_CURATED:
            warnings.append(
                "No SQL warehouse and no `VECTOR_SEARCH_INDEX_CURATED`: "
                "AyurGenix structured retrieval is PDF-only unless you configure one of them."
            )
        elif not config.SQL_WAREHOUSE_CONFIGURED and config.VECTOR_SEARCH_INDEX_CURATED:
            warnings.append(
                "No SQL warehouse: using curated **Vector Search** + PDF chunks (warehouse-free path)."
            )

        try:
            curated_sql_rows = search_curated_sql(msg)
        except Exception as e:
            warnings.append(f"SQL curated search failed: {e}")
            curated_sql_rows = local_retrieval.search_curated_local(msg, config.CURATED_SQL_LIMIT)

    ctx = _build_context(pdf_hits, curated_vec_hits, curated_sql_rows)
    answer = sarvam_llm.generate_answer(sarvam_llm.SYSTEM_DEFAULT, ctx, msg)
    return {
        "answer": answer,
        "chunk_hits": pdf_hits,
        "curated_vector_hits": curated_vec_hits,
        "curated_hits": curated_sql_rows,
        "warnings": warnings,
        "context_preview": ctx[:2500],
    }


def legacy_symptom_card(user_message: str) -> dict[str, Any]:
    """Shape compatible with the old Streamlit 'disease / dosha / advice' demo."""
    r = run_chat(user_message)
    vec_rows = r.get("curated_vector_hits") or []
    sql_rows = r.get("curated_hits") or []
    row = vec_rows[0] if vec_rows else (sql_rows[0] if sql_rows else None)
    if not row:
        ans = (r.get("answer") or "Try a longer symptom description.").strip()
        return {
            "disease": "No close match",
            "dosha": ["—"],
            "advice": ans[:4000],
            "warnings": r.get("warnings"),
        }
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
