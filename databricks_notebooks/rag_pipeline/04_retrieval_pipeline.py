# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Retrieval Pipeline
# MAGIC
# MAGIC Flow:
# MAGIC 1. Accept user query
# MAGIC 2. Convert query to embedding
# MAGIC 3. Retrieve top-k similar chunks from Vector Search index
# MAGIC 4. Return `chunk_text` and similarity score

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

TOP_K = 5
USER_QUERY = "What are the Ayurvedic recommendations for digestion and immunity?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build query embedding

# COMMAND ----------

# DBTITLE 1,Cell 4
from pyspark.sql import functions as F

try:
    # Escape single quotes for SQL by replacing with double single quotes
    escaped_query = USER_QUERY.replace("'", "''")
    query_embedding_row = spark.sql(
        f"""
        SELECT ai_query('{EMBEDDING_ENDPOINT}', '{escaped_query}') AS embedding
        """
    ).first()
    query_embedding = query_embedding_row["embedding"]
except Exception as exc:
    raise RuntimeError(
        "Query embedding generation failed. Check embedding endpoint and permissions."
    ) from exc

if not query_embedding:
    raise ValueError("Empty query embedding returned.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve similar chunks

# COMMAND ----------

# DBTITLE 1,Install Vector Search package
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

try:
    from databricks.vector_search.client import VectorSearchClient
except Exception as exc:
    raise RuntimeError("Vector Search client unavailable.") from exc

vsc = VectorSearchClient()

try:
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT_NAME,
        index_name=VECTOR_INDEX_NAME,
    )
    response = index.similarity_search(
        query_vector=query_embedding,
        columns=["chunk_text", "source_file", "page_number"],
        num_results=TOP_K,
    )
except Exception as exc:
    raise RuntimeError("Vector similarity search failed.") from exc

results = response.get("result", {}).get("data_array", [])
if not results:
    print("No retrieval results found.")
else:
    print("Top retrieval results:")
    for row in results:
        # Returned format generally: [chunk_text, source_file, page_number, score]
        chunk_text = row[0] if len(row) > 0 else None
        score = row[-1] if len(row) > 0 else None
        print({"chunk_text": chunk_text, "similarity_score": score})

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name="ayurgenix-vs-endpoint", 
                      index_name="bricksiitm.ayurgenix.knowledge_base_embeddings_index")
print(index.describe())

# COMMAND ----------

# DBTITLE 1,Monitor index sync progress
from databricks.vector_search.client import VectorSearchClient
import time

vsc = VectorSearchClient()

print(f"Monitoring index: {VECTOR_INDEX_NAME}")
print(f"Expected row count: 446\n")
print("=" * 80)

# Get current status
index = vsc.get_index(
    endpoint_name=VECTOR_ENDPOINT_NAME,
    index_name=VECTOR_INDEX_NAME
)

status = index.describe()['status']

print(f"\n📊 Current Status:")
print(f"  State: {status['detailed_state']}")
print(f"  Ready: {status['ready']}")
print(f"  Indexed Rows: {status.get('indexed_row_count', 0)} / 446")
print(f"  Message: {status.get('message', 'N/A')}")

if status['ready']:
    print("\n✅ Index is ONLINE and ready for queries!")
else:
    print("\n⏳ Index is still provisioning. Re-run this cell to check progress.")
    print("\nStage progression:")
    print("  1. PROVISIONING_ENDPOINT (endpoint setup)")
    print("  2. PROVISIONING_PIPELINE_RESOURCES (pipeline setup) ← Check here")
    print("  3. SYNCING (data ingestion - row count increases)")
    print("  4. ONLINE (ready for queries)")

print("\n" + "=" * 80)

# COMMAND ----------


