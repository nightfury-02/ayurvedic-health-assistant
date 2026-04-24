# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Setup Catalog, Schema, and Volume
# MAGIC This notebook creates Unity Catalog objects for the AyurGenixAI ingestion pipeline.
# MAGIC
# MAGIC **Objects**
# MAGIC - Catalog: `bricksiitm`
# MAGIC - Schema: `ayurgenix`
# MAGIC - Volume: `files`

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
VOLUME = "files"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

print(f"Catalog ready: {CATALOG}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")
print(f"Volume ready: {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify volume path
# MAGIC Expected raw data directory:
# MAGIC `/Volumes/bricksiitm/ayurgenix/files/raw_data/`

# COMMAND ----------

RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"
print(f"Raw data path configured: {RAW_DATA_PATH}")

# COMMAND ----------


