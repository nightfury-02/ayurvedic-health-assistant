# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Validate and Query
# MAGIC This notebook validates the final table and runs basic source-wise queries.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
FINAL_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base"

# COMMAND ----------

df = spark.table(FINAL_TABLE)

row_count = df.count()
if row_count == 0:
    raise ValueError(f"Table {FINAL_TABLE} exists but has zero rows.")

print(f"Total rows in {FINAL_TABLE}: {row_count}")
print(f"Schema: {df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample rows

# COMMAND ----------

display(df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Row counts by source_type

# COMMAND ----------

display(
    df.groupBy("source_type")
    .agg(F.count("*").alias("rows"))
    .orderBy("source_type")
)

# COMMAND ----------

display(
    df.select(
        F.max(F.length("chunk_text")).alias("max_len"),
        F.min(F.length("chunk_text")).alias("min_len"),
        F.avg(F.length("chunk_text")).cast("int").alias("avg_len"),
        F.expr("percentile_approx(length(chunk_text), 0.5)").alias("median_len"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick SQL checks

# COMMAND ----------

spark.sql(
    f"""
    SELECT source_type, COUNT(*) AS rows
    FROM {FINAL_TABLE}
    GROUP BY source_type
    ORDER BY source_type
    """
).show()

# COMMAND ----------

spark.sql(
    f"""
    SELECT
      MAX(length(chunk_text)) AS max_len,
      MIN(length(chunk_text)) AS min_len,
      AVG(length(chunk_text)) AS avg_len
    FROM {FINAL_TABLE}
    """
).show()

# COMMAND ----------

