# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Generate Embeddings
# MAGIC
# MAGIC This notebook reads `processed_knowledge_base`, generates embeddings for `chunk_text`,
# MAGIC and writes output to `ayurveda_assistant.ingestion.knowledge_base_embeddings`.
# MAGIC
# MAGIC Default strategy uses Databricks Model Serving embedding endpoint (`ai_query`).
# MAGIC A fallback using SentenceTransformers is provided.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

CATALOG = "ayurveda_assistant"
SCHEMA = "ingestion"

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.processed_knowledge_base"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings"

# Update to your actual embedding endpoint name if needed.
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
USE_SENTENCE_TRANSFORMERS_FALLBACK = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load processed knowledge base

# COMMAND ----------

try:
    kb_df = spark.table(SOURCE_TABLE)
except Exception as exc:
    raise RuntimeError(f"Cannot load source table: {SOURCE_TABLE}") from exc

required_cols = [
    "row_id",
    "chunk_id",
    "chunk_index",
    "chunk_text",
    "source_type",
    "source_file",
    "page_number",
]
missing_cols = [c for c in required_cols if c not in kb_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in source table: {missing_cols}")

if kb_df.count() == 0:
    raise ValueError(f"Source table is empty: {SOURCE_TABLE}")

null_chunk_count = kb_df.filter(F.col("chunk_text").isNull() | (F.length(F.trim(F.col("chunk_text"))) == 0)).count()
if null_chunk_count > 0:
    raise ValueError(f"Found {null_chunk_count} empty chunk_text rows. Fix ingestion before embedding generation.")

print(f"[DEBUG][EMBED] Source rows: {kb_df.count()}")
print("[DEBUG][EMBED] Sample chunk_text values:")
display(kb_df.select("chunk_id", "chunk_text").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate embeddings

# COMMAND ----------

if not USE_SENTENCE_TRANSFORMERS_FALLBACK:
    try:
        embedded_df = kb_df.withColumn(
            "embedding",
            F.expr(f"ai_query('{EMBEDDING_ENDPOINT}', chunk_text)"),
        )
    except Exception as exc:
        raise RuntimeError(
            "Databricks embedding generation failed. "
            "Check endpoint name/permissions or enable SentenceTransformers fallback."
        ) from exc
else:
    # Fallback path: useful where ai_query is unavailable.
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encode_udf = F.udf(
        lambda text: [float(x) for x in model.encode(text or "").tolist()],
        T.ArrayType(T.FloatType()),
    )
    embedded_df = kb_df.withColumn("embedding", encode_udf(F.col("chunk_text")))

embedded_df = embedded_df.withColumn("embedding", F.col("embedding").cast("array<float>"))
embedded_df = embedded_df.withColumn("embedding_dim", F.size(F.col("embedding")))

empty_embedding_count = embedded_df.filter(F.col("embedding").isNull() | (F.col("embedding_dim") == 0)).count()
if empty_embedding_count > 0:
    raise ValueError(f"Generated {empty_embedding_count} empty embeddings. Check endpoint behavior.")

print("[DEBUG][EMBED] Embedding dimension distribution:")
display(embedded_df.groupBy("embedding_dim").count().orderBy(F.col("count").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save embeddings table

# COMMAND ----------

try:
    (
        embedded_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TARGET_TABLE)
    )
except Exception as exc:
    raise RuntimeError(f"Failed writing embeddings table: {TARGET_TABLE}") from exc

print(f"Saved embeddings table: {TARGET_TABLE}")
display(embedded_df.select("chunk_id", "chunk_text", "embedding_dim").limit(5))

# COMMAND ----------


