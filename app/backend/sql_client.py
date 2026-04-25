"""Run SQL against a Databricks SQL warehouse."""

from __future__ import annotations

from typing import Any

from . import config


def _server_hostname() -> str:
    h = config.DATABRICKS_HOST.replace("https://", "").replace("http://", "")
    return h.split("/")[0].strip()


def run_sql(sql: str) -> list[dict[str, Any]]:
    if config.USE_LOCAL_FALLBACK:
        raise RuntimeError("run_sql is disabled (USE_LOCAL_FALLBACK is on).")
    if not config.SQL_WAREHOUSE_CONFIGURED:
        raise RuntimeError(
            "DATABRICKS_HTTP_PATH is not set — configure a SQL warehouse path for curated SQL, "
            "or omit it and use Vector Search–only RAG."
        )
    from databricks.sql import connect

    conn = connect(
        server_hostname=_server_hostname(),
        http_path=config.DATABRICKS_HTTP_PATH,
        access_token=config.DATABRICKS_TOKEN,
    )
    try:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchall()
            return [dict(zip(cols, row)) for row in rows]
        finally:
            cur.close()
    finally:
        conn.close()
