"""Environment settings: Databricks SQL, Vector Search, FM embeddings, Sarvam."""

from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _workspace_host() -> str:
    h = (os.getenv("DATABRICKS_HOST") or "").strip().rstrip("/")
    if not h:
        return ""
    if h.startswith("http://"):
        return "https://" + h[len("http://") :]
    if not h.startswith("https://"):
        return "https://" + h
    return h


DATABRICKS_HOST = _workspace_host()
DATABRICKS_TOKEN = (os.getenv("DATABRICKS_TOKEN") or "").strip()
DATABRICKS_HTTP_PATH = (os.getenv("DATABRICKS_HTTP_PATH") or "").strip()

CATALOG = (os.getenv("AYURVEDA_CATALOG") or "ayurveda_lakehouse").strip()
SCHEMA = (os.getenv("AYURVEDA_SCHEMA") or "ayurgenix").strip()

CURATED_TABLE = os.getenv("AYURVEDA_CURATED_TABLE", f"{CATALOG}.{SCHEMA}.ayurgenix_curated")
CHUNKS_TABLE = os.getenv("AYURVEDA_CHUNKS_TABLE", f"{CATALOG}.{SCHEMA}.pdf_text_chunks")

VECTOR_SEARCH_ENDPOINT = (os.getenv("VECTOR_SEARCH_ENDPOINT") or "").strip()
VECTOR_SEARCH_INDEX = (os.getenv("VECTOR_SEARCH_INDEX") or "").strip()
VECTOR_SEARCH_COLUMNS = [
    c.strip()
    for c in (os.getenv("VECTOR_SEARCH_COLUMNS") or "").split(",")
    if c.strip()
] or ["chunk_pk", "chunk_text", "source_file", "page_number", "chunk_index"]

EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "databricks-qwen3-embedding-0-6b").strip()

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY") or "").strip()
SARVAM_MODEL = (os.getenv("SARVAM_MODEL") or "sarvam-30b").strip()

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CSV_PATH = Path(os.getenv("AYUR_LOCAL_CSV", str(_REPO_ROOT / "raw_data" / "AyurGenixAI_Dataset.csv")))

USE_LOCAL_FALLBACK = _bool("USE_LOCAL_FALLBACK") or not (
    DATABRICKS_TOKEN and DATABRICKS_HTTP_PATH and DATABRICKS_HOST
)

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
CURATED_SQL_LIMIT = int(os.getenv("CURATED_SQL_LIMIT", "5"))
