# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Validate and Query
# MAGIC This notebook validates the final table and runs basic source-wise queries.

# COMMAND ----------

from pyspark.sql import functions as F

FINAL_TABLE = "bricksiitm.ayurgenix.knowledge_base"

df = spark.table(FINAL_TABLE)

row_count = df.count()
if row_count == 0:
    raise ValueError(f"Table {FINAL_TABLE} exists but has zero rows.")

print(f"Total rows in {FINAL_TABLE}: {row_count}")

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
