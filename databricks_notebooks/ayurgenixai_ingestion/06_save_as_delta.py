# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Save Unified Dataset as Delta
# MAGIC This notebook persists the unified RAG dataset into the final Delta table:
# MAGIC `bricksiitm.ayurgenix.knowledge_base`

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
UNIFIED_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base_unified_staging"
FINAL_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base"

unified_df = spark.table(UNIFIED_STAGING_TABLE)

if unified_df.count() == 0:
    raise ValueError("Cannot save final table because unified staging table is empty.")

(
    unified_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FINAL_TABLE)
)

print(f"Saved final Delta table: {FINAL_TABLE}")
