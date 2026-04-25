"""Structured AyurGenix retrieval via SQL (glue, not primary semantic path)."""

from __future__ import annotations

import re
from typing import Any

from . import config, schema, sql_client


def _tokens(q: str) -> list[str]:
    q = (q or "").strip()
    parts = re.split(r"[^\w]+", q, flags=re.UNICODE)
    return [t.lower() for t in parts if len(t) > 2][:8]


def search_curated_sql(query: str) -> list[dict[str, Any]]:
    if config.USE_LOCAL_FALLBACK:
        return []
    tokens = _tokens(query)
    if not tokens:
        tokens = [query.lower()[:64]] if query.strip() else ["a"]
    concat_expr = "concat_ws(' ', " + ", ".join(schema.CURATED_TEXT_COLUMNS) + ")"
    clauses: list[str] = []
    for t in tokens:
        lit = t.replace("'", "''")
        clauses.append(f"lower({concat_expr}) LIKE lower('%{lit}%')")
    where = "(" + " OR ".join(clauses) + ")"
    cols = ", ".join(schema.CURATED_DISPLAY_COLUMNS)
    sql = f"SELECT {cols} FROM {config.CURATED_TABLE} WHERE {where} LIMIT {int(config.CURATED_SQL_LIMIT)}"
    return sql_client.run_sql(sql)
