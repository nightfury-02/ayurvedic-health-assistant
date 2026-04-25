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

# DBTITLE 1,Enable Change Data Feed on source table
# Enable Change Data Feed (CDF) on the source table
# This is required for Delta Sync indexes to track changes

print(f"Enabling Change Data Feed on {EMBEDDINGS_TABLE}...")

spark.sql(f"""
    ALTER TABLE {EMBEDDINGS_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print("✅ Change Data Feed enabled successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create endpoint and index

# COMMAND ----------

# DBTITLE 1,Install Vector Search package
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 5
# Initialize client
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

print(f"Using existing endpoint: {VECTOR_ENDPOINT_NAME}")

# Create or validate index
try:
    indexes = vsc.list_indexes(VECTOR_ENDPOINT_NAME)

    # Handle response safely
    index_list = (
        indexes.get("vector_indexes")
        or indexes.get("indexes")
        or []
    )

    index_names = {idx.get("name") for idx in index_list}

    if VECTOR_INDEX_NAME not in index_names:
        print(f"Creating index: {VECTOR_INDEX_NAME}")

        try:
            vsc.create_delta_sync_index(
                endpoint_name=VECTOR_ENDPOINT_NAME,
                index_name=VECTOR_INDEX_NAME,
                source_table_name=EMBEDDINGS_TABLE,
                pipeline_type="TRIGGERED",
                primary_key=PRIMARY_KEY,
                embedding_dimension=EMBEDDING_DIMENSION,
                embedding_vector_column=EMBEDDING_COLUMN,
            )
            print("Index creation triggered. It may take a few minutes...")
        except Exception as create_exc:
            if "already exists" in str(create_exc):
                print(f"Index already exists: {VECTOR_INDEX_NAME}")
            else:
                raise

    else:
        print(f"Index already exists: {VECTOR_INDEX_NAME}")

except Exception as exc:
    print("Detailed error:", str(exc))
    raise RuntimeError("Failed to create or validate Vector Search index.") from exc


print("✅ Vector Search setup complete.")
print(f"Endpoint: {VECTOR_ENDPOINT_NAME}")
print(f"Index: {VECTOR_INDEX_NAME}")

# COMMAND ----------



