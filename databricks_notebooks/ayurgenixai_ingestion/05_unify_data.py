# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Build Processed Knowledge Base
# MAGIC This notebook unifies CSV/PDF staging tables and applies robust preprocessing:
# MAGIC - context-aware chunking on `raw_text`
# MAGIC - source metadata preservation
# MAGIC - deterministic chunk metadata
# MAGIC
# MAGIC Final output table:
# MAGIC `bricksiitm.ayurgenix.processed_knowledge_base`

# COMMAND ----------

from typing import List
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

CSV_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.csv_chunks_staging"
PDF_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.pdf_chunks_staging"
PROCESSED_TABLE = f"{CATALOG}.{SCHEMA}.processed_knowledge_base"

# Word-based chunking defaults (required range: 300-500, overlap 50-100)
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 80


def chunk_text_by_words(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    if text is None:
        return []

    normalized = " ".join(text.split())
    if not normalized:
        return []

    words = normalized.split(" ")
    if len(words) <= chunk_size:
        # Do not split small text unnecessarily.
        return [normalized]

    chunks = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step
    return chunks


chunker_udf = F.udf(chunk_text_by_words, T.ArrayType(T.StringType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and validate staging tables

# COMMAND ----------

try:
    csv_df = spark.table(CSV_STAGING_TABLE)
    pdf_df = spark.table(PDF_STAGING_TABLE)
except Exception as exc:
    raise RuntimeError("Unable to read CSV/PDF staging tables. Run ingestion notebooks first.") from exc

required_cols = ["text_chunk", "source_type", "metadata"]
for col_name in required_cols:
    if col_name not in csv_df.columns:
        raise ValueError(f"Column missing in CSV staging table: {col_name}")
    if col_name not in pdf_df.columns:
        raise ValueError(f"Column missing in PDF staging table: {col_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardize metadata and construct raw_text context

# COMMAND ----------

csv_std = (
    csv_df.select("text_chunk", "source_type", "metadata")
    .withColumn("source_file", F.get_json_object(F.col("metadata"), "$.file_name"))
    .withColumn("page_number", F.lit(None).cast("int"))
)

pdf_std = (
    pdf_df.select("text_chunk", "source_type", "metadata")
    .withColumn("source_file", F.get_json_object(F.col("metadata"), "$.file_name"))
    .withColumn("page_number", F.get_json_object(F.col("metadata"), "$.page_number").cast("int"))
)

base_df = csv_std.unionByName(pdf_std, allowMissingColumns=True)

if base_df.count() == 0:
    raise ValueError("Unified source rows are empty. Check CSV/PDF staging tables.")

base_df = (
    base_df.withColumn("source_file", F.coalesce(F.col("source_file"), F.lit("unknown_file")))
    .withColumn("row_id", F.expr("uuid()"))
    .withColumn(
        "raw_text",
        F.concat(
            F.lit("Source: "),
            F.col("source_type"),
            F.lit(" | File: "),
            F.col("source_file"),
            F.when(
                F.col("page_number").isNotNull(),
                F.concat(F.lit(" | Page: "), F.col("page_number").cast("string")),
            ).otherwise(F.lit("")),
            F.lit(" | Content: "),
            F.coalesce(F.col("text_chunk"), F.lit("")),
        ),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk `raw_text`, explode chunks, and build chunk metadata

# COMMAND ----------

chunked_df = (
    base_df.withColumn("chunk_array", chunker_udf(F.col("raw_text")))
    .withColumn("chunk_with_pos", F.posexplode_outer(F.col("chunk_array")))
    .select(
        "row_id",
        F.col("chunk_with_pos.pos").alias("chunk_index"),
        F.col("chunk_with_pos.col").alias("chunk_text"),
        "source_type",
        "source_file",
        "page_number",
    )
    .filter(F.col("chunk_text").isNotNull() & (F.length(F.trim(F.col("chunk_text"))) > 0))
)

final_df = (
    chunked_df.withColumn("chunk_index", F.col("chunk_index").cast("int"))
    .withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("row_id"),
                F.col("chunk_index").cast("string"),
                F.col("chunk_text"),
            ),
            256,
        ),
    )
    .select(
        "row_id",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "source_type",
        "source_file",
        F.col("page_number").cast("int").alias("page_number"),
    )
)

if final_df.count() == 0:
    raise ValueError("No chunks produced after context-aware preprocessing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save final processed table

# COMMAND ----------

try:
    (
        final_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PROCESSED_TABLE)
    )
except Exception as exc:
    raise RuntimeError(f"Failed writing Delta table: {PROCESSED_TABLE}") from exc

print(f"Saved processed knowledge base: {PROCESSED_TABLE}")
display(final_df.limit(20))

# COMMAND ----------


