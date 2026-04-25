# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Save Unified Dataset as Delta
# MAGIC This notebook persists the unified RAG dataset into the final Delta table:
# MAGIC `bricksiitm.ayurgenix.knowledge_base`

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

PROCESSED_TABLE = f"{CATALOG}.{SCHEMA}.processed_knowledge_base"
CSV_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.csv_chunks_staging"
PDF_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.pdf_chunks_staging"
FINAL_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base"

# COMMAND ----------



def _table_exists(name: str) -> bool:
    try:
        spark.table(name)
        return True
    except AnalysisException:
        return False


def _from_processed() -> "DataFrame":
    df = spark.table(PROCESSED_TABLE)
    expected = {
        "row_id",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "source_type",
        "source_file",
        "page_number",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(
            f"{PROCESSED_TABLE} is missing expected columns: {sorted(missing)}. "
            "Re-run notebook 05_unify_data.py."
        )
    return df.select(
        "row_id",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "source_type",
        "source_file",
        "page_number",
    )


def _from_staging() -> "DataFrame":
    """Fallback: build a `knowledge_base`-shaped frame from the raw staging tables.

    The staging tables produced by notebooks 03/04 only have
    `id`, `text_chunk`, `source_type`, `metadata`, so the page/source-file
    columns are reconstructed from the JSON `metadata` field where possible.
    """
    parts = []
    for name in (CSV_STAGING_TABLE, PDF_STAGING_TABLE):
        if not _table_exists(name):
            continue
        df = spark.table(name)
        meta = F.from_json(
            F.col("metadata"),
            "struct<file_name:string,page_count:int,record_type:string,chunk_index:int>",
        )
        parts.append(
            df.withColumn("meta", meta)
            .select(
                F.expr("uuid()").alias("row_id"),
                F.col("id").alias("chunk_id"),
                F.col("meta.chunk_index").alias("chunk_index"),
                F.col("text_chunk").alias("chunk_text"),
                "source_type",
                F.col("meta.file_name").alias("source_file"),
                F.lit(None).cast("int").alias("page_number"),
            )
        )

    if not parts:
        raise FileNotFoundError(
            "Neither processed_knowledge_base nor any *_chunks_staging table "
            "exists. Run notebooks 03/04/05 first."
        )

    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p, allowMissingColumns=True)
    return out


# COMMAND ----------


if _table_exists(PROCESSED_TABLE):
    print(f"Sourcing from: {PROCESSED_TABLE}")
    final_df = _from_processed()
else:
    print(
        f"{PROCESSED_TABLE} not found. Falling back to staging tables "
        f"({CSV_STAGING_TABLE}, {PDF_STAGING_TABLE})."
    )
    final_df = _from_staging()

row_count = final_df.count()
if row_count == 0:
    raise ValueError("Source dataset is empty; refusing to overwrite knowledge_base.")

print(f"Source rows: {row_count}")

# COMMAND ----------

(
    final_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FINAL_TABLE)
)

print(f"Saved final Delta table: {FINAL_TABLE} ({row_count} rows)")

# COMMAND ----------

display(spark.table(FINAL_TABLE).limit(10))
display(
    spark.table(FINAL_TABLE)
    .groupBy("source_type")
    .agg(F.count("*").alias("rows"))
    .orderBy("source_type")
)