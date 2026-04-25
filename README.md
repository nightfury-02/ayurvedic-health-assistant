# Ayurvedic Health Assistant (Databricks)

Lakehouse ingestion (AyurGenix Delta + PDF chunks) and a **Databricks App** (Streamlit) that runs **hybrid RAG**: **Mosaic AI Vector Search** on PDF chunks, **SQL** on `ayurgenix_curated`, and **Sarvam** for grounded answers.

## Ingestion order (notebooks)

Run on a Unity Catalog–enabled workspace (Serverless or UC cluster):

1. [`data_ingestion/00_setup_catalog_schema_volume.ipynb`](data_ingestion/00_setup_catalog_schema_volume.ipynb)
2. [`data_ingestion/01_ingest_ayurgenix_dataset.ipynb`](data_ingestion/01_ingest_ayurgenix_dataset.ipynb)
3. [`data_ingestion/02_prepare_ayurgenix_curated.ipynb`](data_ingestion/02_prepare_ayurgenix_curated.ipynb)
4. [`data_ingestion/03_register_pdf_sources_and_explore.ipynb`](data_ingestion/03_register_pdf_sources_and_explore.ipynb)
5. [`data_ingestion/04_create_chatbot_analytics_views.ipynb`](data_ingestion/04_create_chatbot_analytics_views.ipynb)
6. [`data_ingestion/05_prepare_vector_search_chunks.ipynb`](data_ingestion/05_prepare_vector_search_chunks.ipynb) — builds `pdf_chunks_for_vector` and documents Vector Search index creation + sync  
7. [`data_ingestion/06_curated_rows_for_vector.ipynb`](data_ingestion/06_curated_rows_for_vector.ipynb) — builds **`curated_rows_for_vector`** with **symptom-weighted** `embedding_text` for a **second** Vector Search index (warehouse-free AyurGenix retrieval)

## Databricks App (Streamlit)

App code lives under [`app/`](app/). Deploy as a [Databricks App](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html) with root `app/` (this repo’s layout matches `streamlit run app.py` via [`app/app.yaml`](app/app.yaml)).

Install dependencies from [`requirements.txt`](requirements.txt) (Apps pick up `requirements.txt` next to the entry module).

### Environment variables (set as App secrets / env)

| Variable | Purpose |
|----------|---------|
| `DATABRICKS_HOST` | Workspace URL, e.g. `https://adb-....databricks.com` |
| `DATABRICKS_TOKEN` | PAT with access to SQL warehouse + Vector Search + UC tables |
| `DATABRICKS_HTTP_PATH` | **Optional.** SQL warehouse HTTP path (e.g. `/sql/1.0/warehouses/xxxx`). If omitted, the app still runs **Vector Search + Sarvam** but skips **AyurGenix curated SQL** retrieval. |
| `SARVAM_API_KEY` | Sarvam API subscription key for chat completions |
| `VECTOR_SEARCH_ENDPOINT` | Vector Search **endpoint name** (not a URL); same endpoint can host **both** indexes |
| `VECTOR_SEARCH_INDEX` or `VECTOR_SEARCH_INDEX_PDF` | Full UC name of the **PDF chunks** index (notebook **05**) |
| `VECTOR_SEARCH_INDEX_CURATED` | Full UC name of the **AyurGenix curated** index (notebook **06**); optional but recommended without a SQL warehouse |
| `VECTOR_SEARCH_COLUMNS` / `VECTOR_SEARCH_COLUMNS_PDF` | Optional; defaults match notebook **05** |
| `VECTOR_SEARCH_COLUMNS_CURATED` | Optional; defaults match notebook **06** smoke test (excludes `embedding_text` from query payload) |
| `VECTOR_TOP_K_PDF` / `VECTOR_TOP_K_CURATED` | Optional result counts (defaults **8** / **10** — extra headroom for symptom retrieval on curated) |
| `SARVAM_MODEL` | Optional; default `sarvam-30b` |
| `EMBEDDING_MODEL` | Optional; used only if you call [`app/backend/embeddings.py`](app/backend/embeddings.py) for custom batch jobs (default `databricks-qwen3-embedding-0-6b`) |
| `AYURVEDA_CATALOG` / `AYURVEDA_SCHEMA` | Optional overrides (default `ayurveda_lakehouse` / `ayurgenix`) |
| `USE_LOCAL_FALLBACK` | Set to `1` only for offline demos with local CSV; leave **unset** on Databricks |

If `DATABRICKS_HOST` / `DATABRICKS_TOKEN` are not set, the app uses **local CSV** lexical retrieval from `raw_data/AyurGenixAI_Dataset.csv` (no Vector Search).

**Without a SQL warehouse:** omit `DATABRICKS_HTTP_PATH`. Configure **both** Vector Search indexes (PDF + **`VECTOR_SEARCH_INDEX_CURATED`** from notebook **06**) and Sarvam. Curated rows use **symptom-heavy** `embedding_text` so user symptom queries match strongly against the CSV layer.

### Backend layout

| Module | Role |
|--------|------|
| [`app/backend/rag.py`](app/backend/rag.py) | Orchestrates retrieval + Sarvam |
| [`app/backend/vector_retrieval.py`](app/backend/vector_retrieval.py) | `similarity_search` on **PDF** and **curated** Vector Search indexes |
| [`app/backend/curated_sql.py`](app/backend/curated_sql.py) | Token OR-query over concatenated curated text columns |
| [`app/backend/sql_client.py`](app/backend/sql_client.py) | `databricks-sql-connector` to the warehouse |
| [`app/backend/sarvam_llm.py`](app/backend/sarvam_llm.py) | Sarvam chat with strict grounding prompt |
| [`app/backend/embeddings.py`](app/backend/embeddings.py) | Optional: Foundation Model **Embeddings** API (OpenAI-compatible client) for your own batch pipelines |

## Disclaimer

Outputs are for **education and wellness information only**, not a substitute for professional medical advice.
