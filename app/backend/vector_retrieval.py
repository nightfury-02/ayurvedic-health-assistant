"""Mosaic AI Vector Search: semantic chunk retrieval."""

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


def search_chunks(query_text: str) -> list[dict[str, Any]]:
    if not config.VECTOR_SEARCH_ENDPOINT or not config.VECTOR_SEARCH_INDEX:
        return []
    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError:
        return []
    try:
        vsc = VectorSearchClient(disable_notice=True)
        index = vsc.get_index(
            endpoint_name=config.VECTOR_SEARCH_ENDPOINT,
            index_name=config.VECTOR_SEARCH_INDEX,
        )
        raw = index.similarity_search(
            query_text=query_text,
            columns=config.VECTOR_SEARCH_COLUMNS,
            num_results=config.VECTOR_TOP_K,
            query_type="HYBRID",
        )
        return _normalize(raw)
    except Exception:
        return []
