"""Mosaic AI Vector Search: PDF chunks + optional second index on AyurGenix curated rows."""

from __future__ import annotations

from typing import Any

from . import config


def _normalize(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    if "result" in raw and isinstance(raw["result"], dict):
        r = raw["result"]
        cols = r.get("columns") or r.get("column_names")
        rows = r.get("data_array") or r.get("rows") or []
        if cols and rows:
            return [dict(zip(cols, row)) for row in rows]
    r2 = raw.get("manifest") or raw
    cols = r2.get("columns") or r2.get("column_names")
    rows = r2.get("data_array") or r2.get("rows") or []
    if cols and rows:
        return [dict(zip(cols, row)) for row in rows]
    data = raw.get("data") or raw.get("result")
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _similarity_search(
    index_name: str,
    columns: list[str],
    query_text: str,
    num_results: int,
) -> list[dict[str, Any]]:
    if not config.VECTOR_SEARCH_ENDPOINT or not index_name:
        return []
    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError:
        return []
    try:
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=config.VECTOR_SEARCH_ENDPOINT,
            index_name=index_name,
        )
        raw = index.similarity_search(
            query_text=query_text,
            columns=columns,
            num_results=num_results,
            query_type="HYBRID",
        )
        return _normalize(raw)
    except Exception:
        return []


def search_pdf_chunks(query_text: str) -> list[dict[str, Any]]:
    """Primary PDF chunk index (same as legacy `search_chunks`)."""
    return _similarity_search(
        config.VECTOR_SEARCH_INDEX_PDF,
        config.VECTOR_SEARCH_COLUMNS_PDF,
        query_text,
        config.VECTOR_TOP_K_PDF,
    )


def search_curated_vector(query_text: str) -> list[dict[str, Any]]:
    """Second index: symptom-weighted AyurGenix rows (`curated_rows_for_vector`)."""
    return _similarity_search(
        config.VECTOR_SEARCH_INDEX_CURATED,
        config.VECTOR_SEARCH_COLUMNS_CURATED,
        query_text,
        config.VECTOR_TOP_K_CURATED,
    )


def search_chunks(query_text: str) -> list[dict[str, Any]]:
    """Backward-compatible alias for PDF chunk search."""
    return search_pdf_chunks(query_text)
