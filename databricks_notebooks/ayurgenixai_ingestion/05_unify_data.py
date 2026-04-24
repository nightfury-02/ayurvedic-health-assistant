# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Unify CSV and PDF Data
# MAGIC This notebook reads CSV/PDF staging Delta tables and creates a unified RAG-ready DataFrame.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

CSV_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.csv_chunks_staging"
PDF_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.pdf_chunks_staging"
UNIFIED_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base_unified_staging"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read staging tables

# COMMAND ----------

csv_df = spark.table(CSV_STAGING_TABLE)
pdf_df = spark.table(PDF_STAGING_TABLE)

required_cols = ["id", "text_chunk", "source_type", "metadata"]
for col_name in required_cols:
    if col_name not in csv_df.columns:
        raise ValueError(f"Column missing in CSV staging table: {col_name}")
    if col_name not in pdf_df.columns:
        raise ValueError(f"Column missing in PDF staging table: {col_name}")

csv_std = csv_df.select(required_cols)
pdf_std = pdf_df.select(required_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union sources and re-key ids

# COMMAND ----------

unified_df = csv_std.unionByName(pdf_std, allowMissingColumns=True)

if unified_df.count() == 0:
    raise ValueError("Unified DataFrame is empty. Check CSV/PDF staging inputs.")

# Reassign continuous chunk ids across all sources.
window_spec = Window.orderBy(F.monotonically_increasing_id())
unified_df = (
    unified_df.withColumn("chunk_id", F.row_number().over(window_spec))
    .drop("id")
    .withColumnRenamed("chunk_id", "id")
    .withColumn("id", F.col("id").cast("string"))
)

(
    unified_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(UNIFIED_STAGING_TABLE)
)

print(f"Saved unified staging table: {UNIFIED_STAGING_TABLE}")
display(unified_df.limit(20))
