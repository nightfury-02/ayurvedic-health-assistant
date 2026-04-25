# Ayurvedic Health Assistant

A small reference project that pairs a lightweight Streamlit demo with a
Databricks-based RAG pipeline over Ayurvedic CSV/PDF sources.

> Educational use only. Not medical advice.

## Layout

```
app/                          # Streamlit demo + simple rule-based matcher
  app.py
  app.yaml                    # Databricks Apps run config
  backend/test_match.py
databricks_notebooks/
  ayurgenixai_ingestion/      # Catalog/volume setup + CSV/PDF ingestion
  ayurgenixai_preprocessing/  # Cleaning + chunking pipeline (notebook)
  rag_pipeline/               # Embeddings, vector index, retrieval, RAG, API
raw_data/                     # Sample CSV + PDFs used for ingestion
```

## Run the Streamlit demo locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open the URL Streamlit prints (typically <http://localhost:8501>).

## Databricks pipeline (high level)

Run notebooks in order inside a Databricks workspace with Unity Catalog:

1. `ayurgenixai_ingestion/01_setup_catalog_and_volume.py`
2. `ayurgenixai_ingestion/02_validate_raw_files.py`
3. `ayurgenixai_ingestion/03_ingest_csv.py` and `04_ingest_pdfs.py`
4. `ayurgenixai_ingestion/05_unify_data.py`
5. `ayurgenixai_ingestion/06_save_as_delta.py` then `07_validate_and_query.py`
6. `rag_pipeline/02_generate_embeddings.py`
7. `rag_pipeline/03_create_vector_index.py`
8. `rag_pipeline/04_retrieval_pipeline.py`
9. `rag_pipeline/05_rag_pipeline.py` for end-to-end RAG
10. `rag_pipeline/06_api_serving.py` for a FastAPI wrapper

Update the `CATALOG`, `SCHEMA`, endpoint, and index names at the top of each
notebook to match your workspace.
