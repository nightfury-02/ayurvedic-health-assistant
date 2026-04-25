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

from pyspark.sql import functions as F

try:
    query_embedding_row = spark.sql(
        f"""
        SELECT ai_query('{EMBEDDING_ENDPOINT}', '{USER_QUERY.replace("'", "\\'")}') AS embedding
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
