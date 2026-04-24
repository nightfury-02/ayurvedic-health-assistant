# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Validate Raw Files
# MAGIC This notebook lists all files under the raw Unity Catalog volume path and separates CSV and PDF inputs.

# COMMAND ----------

from pyspark.sql import functions as F

RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"


def discover_files(path: str):
    """
    Discover files from a Unity Catalog volume path.
    Tries dbutils.fs.ls first, then falls back to Spark binaryFile listing.
    """
    try:
        listed = dbutils.fs.ls(path)
        return [f.path for f in listed if not f.isDir()]
    except Exception as exc:
        print(f"dbutils.fs.ls failed for {path}. Falling back to Spark listing. Error: {str(exc)}")
        try:
            files_df = (
                spark.read.format("binaryFile")
                .option("recursiveFileLookup", "true")
                .load(path)
                .select(F.col("path"))
                .distinct()
            )
            return [r.path for r in files_df.collect()]
        except Exception as fallback_exc:
            raise RuntimeError(f"Failed to list files in {path}") from fallback_exc


all_paths = discover_files(RAW_DATA_PATH)
if not all_paths:
    raise ValueError(f"No files found in {RAW_DATA_PATH}")

csv_paths = [p for p in all_paths if p.lower().endswith(".csv")]
pdf_paths = [p for p in all_paths if p.lower().endswith(".pdf")]

print(f"Total files: {len(all_paths)}")
print(f"CSV files: {len(csv_paths)}")
print(f"PDF files: {len(pdf_paths)}")

if not csv_paths:
    print("Warning: No CSV files found.")
if not pdf_paths:
    print("Warning: No PDF files found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full CSV Paths

# COMMAND ----------

for p in csv_paths:
    print(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full PDF Paths

# COMMAND ----------

for p in pdf_paths:
    print(p)
