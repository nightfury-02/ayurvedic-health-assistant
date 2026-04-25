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

# PDF chunks index (legacy single env `VECTOR_SEARCH_INDEX` = PDF if `VECTOR_SEARCH_INDEX_PDF` unset)
VECTOR_SEARCH_INDEX_PDF = (
    os.getenv("VECTOR_SEARCH_INDEX_PDF") or os.getenv("VECTOR_SEARCH_INDEX") or ""
).strip()
# Second index: materialised AyurGenix rows (`curated_rows_for_vector`, notebook 06)
VECTOR_SEARCH_INDEX_CURATED = (os.getenv("VECTOR_SEARCH_INDEX_CURATED") or "").strip()

VECTOR_SEARCH_COLUMNS_PDF = [
    c.strip()
    for c in (os.getenv("VECTOR_SEARCH_COLUMNS_PDF") or os.getenv("VECTOR_SEARCH_COLUMNS") or "").split(
        ","
    )
    if c.strip()
] or ["chunk_pk", "chunk_text", "source_file", "page_number", "chunk_index"]

VECTOR_SEARCH_COLUMNS_CURATED = [
    c.strip() for c in (os.getenv("VECTOR_SEARCH_COLUMNS_CURATED") or "").split(",") if c.strip()
] or [
    "curated_pk",
    "Disease",
    "Symptoms",
    "Ayurvedic_Herbs",
    "Formulation",
    "Doshas",
    "Diet_and_Lifestyle_Recommendations",
    "Patient_Recommendations",
    "Herbal_Alternative_Remedies",
    "Diagnosis_Tests",
]

# Backward compatibility for imports of `VECTOR_SEARCH_INDEX` / `VECTOR_SEARCH_COLUMNS`
VECTOR_SEARCH_INDEX = VECTOR_SEARCH_INDEX_PDF
VECTOR_SEARCH_COLUMNS = VECTOR_SEARCH_COLUMNS_PDF

VECTOR_TOP_K_PDF = int(os.getenv("VECTOR_TOP_K_PDF") or os.getenv("VECTOR_TOP_K") or "8")
VECTOR_TOP_K_CURATED = int(os.getenv("VECTOR_TOP_K_CURATED") or os.getenv("VECTOR_TOP_K") or "10")

EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "databricks-qwen3-embedding-0-6b").strip()

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY") or "").strip()
SARVAM_MODEL = (os.getenv("SARVAM_MODEL") or "sarvam-30b").strip()

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CSV_PATH = Path(os.getenv("AYUR_LOCAL_CSV", str(_REPO_ROOT / "raw_data" / "AyurGenixAI_Dataset.csv")))

# Lakehouse auth: workspace host + PAT (used by Vector Search SDK, etc.)
LAKEHOUSE_AUTH_OK = bool(DATABRICKS_TOKEN and DATABRICKS_HOST)

# SQL warehouse is optional: only `curated_sql` needs DATABRICKS_HTTP_PATH.
SQL_WAREHOUSE_CONFIGURED = bool(DATABRICKS_HTTP_PATH)

# Local CSV only when forced or when there is no workspace token (e.g. laptop demo).
USE_LOCAL_FALLBACK = _bool("USE_LOCAL_FALLBACK") or not LAKEHOUSE_AUTH_OK

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
CURATED_SQL_LIMIT = int(os.getenv("CURATED_SQL_LIMIT", "5"))
