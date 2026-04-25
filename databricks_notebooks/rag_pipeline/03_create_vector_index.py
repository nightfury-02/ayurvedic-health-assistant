# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Create Vector Search Endpoint and Index
# MAGIC
# MAGIC This notebook creates:
# MAGIC - Vector Search endpoint
# MAGIC - Delta Sync index on `bricksiitm.ayurgenix.knowledge_base_embeddings`
# MAGIC
# MAGIC Adjust names below once and re-run safely.

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings"
VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"

# Match the embedding dimension of your embedding model.
EMBEDDING_DIMENSION = 1024
EMBEDDING_COLUMN = "embedding"
PRIMARY_KEY = "chunk_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create endpoint and index

# COMMAND ----------

try:
    from databricks.vector_search.client import VectorSearchClient
except Exception as exc:
    raise RuntimeError(
        "Vector Search client is unavailable. Install/enable databricks-vectorsearch."
    ) from exc

vsc = VectorSearchClient()

try:
    endpoints = vsc.list_endpoints()
    endpoint_names = {ep.get("name") for ep in endpoints.get("endpoints", [])}
    if VECTOR_ENDPOINT_NAME not in endpoint_names:
        print(f"Creating Vector Search endpoint: {VECTOR_ENDPOINT_NAME}")
        vsc.create_endpoint(name=VECTOR_ENDPOINT_NAME, endpoint_type="STANDARD")
    else:
        print(f"Endpoint already exists: {VECTOR_ENDPOINT_NAME}")
except Exception as exc:
    raise RuntimeError("Failed to create or validate Vector Search endpoint.") from exc

try:
    indexes = vsc.list_indexes(endpoint_name=VECTOR_ENDPOINT_NAME)
    index_names = {idx.get("name") for idx in indexes.get("vector_indexes", [])}
    if VECTOR_INDEX_NAME not in index_names:
        print(f"Creating index: {VECTOR_INDEX_NAME}")
        vsc.create_delta_sync_index(
            endpoint_name=VECTOR_ENDPOINT_NAME,
            index_name=VECTOR_INDEX_NAME,
            source_table_name=EMBEDDINGS_TABLE,
            pipeline_type="TRIGGERED",
            primary_key=PRIMARY_KEY,
            embedding_dimension=EMBEDDING_DIMENSION,
            embedding_vector_column=EMBEDDING_COLUMN,
        )
    else:
        print(f"Index already exists: {VECTOR_INDEX_NAME}")
except Exception as exc:
    raise RuntimeError("Failed to create or validate Vector Search index.") from exc

print("Vector Search setup complete.")
print(f"Endpoint: {VECTOR_ENDPOINT_NAME}")
print(f"Index: {VECTOR_INDEX_NAME}")
