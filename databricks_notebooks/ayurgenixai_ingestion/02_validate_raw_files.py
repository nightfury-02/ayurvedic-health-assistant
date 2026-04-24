# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Validate Raw Files
# MAGIC This notebook lists all files under the raw Unity Catalog volume path and separates CSV and PDF inputs.

# COMMAND ----------

from pyspark.sql.utils import AnalysisException

RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"


def safe_ls(path: str):
    try:
        return dbutils.fs.ls(path)
    except AnalysisException as exc:
        raise FileNotFoundError(f"Path does not exist or is inaccessible: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to list files in {path}") from exc


files = safe_ls(RAW_DATA_PATH)
if not files:
    raise ValueError(f"No files found in {RAW_DATA_PATH}")

all_paths = [f.path for f in files if not f.isDir()]
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
