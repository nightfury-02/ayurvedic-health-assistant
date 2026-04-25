# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Ingest CSV
# MAGIC This notebook ingests CSV files from Unity Catalog volume storage, applies basic cleanup, creates RAG chunks, and stores a Delta staging table.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import importlib.util
import os

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"
CSV_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.csv_chunks_staging"

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 70


def _import_shared_chunker():
    try:
        from databricks_notebooks.rag_pipeline.chunking_utils import chunk_text_by_words
        return chunk_text_by_words
    except Exception:
        current = os.getcwd()
        while True:
            candidate = os.path.join(
                current, "databricks_notebooks", "rag_pipeline", "chunking_utils.py"
            )
            if os.path.exists(candidate):
                spec = importlib.util.spec_from_file_location("chunking_utils", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.chunk_text_by_words
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        raise ImportError("Unable to locate shared chunking_utils.py")


chunk_text_by_words = _import_shared_chunker()


def text_chunker(text: str):
    return chunk_text_by_words(
        text, chunk_size_words=CHUNK_SIZE_WORDS, overlap_words=CHUNK_OVERLAP_WORDS
    )


chunker_udf = F.udf(text_chunker, "array<string>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detect CSV files

# COMMAND ----------

def discover_files(path: str):
    try:
        listed = dbutils.fs.ls(path)
        return [f.path for f in listed if not f.isDir()]
    except Exception as exc:
        print(f"dbutils.fs.ls failed for {path}. Falling back to Spark listing. Error: {str(exc)}")
        files_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .load(path)
            .select(F.col("path"))
            .distinct()
        )
        return [r.path for r in files_df.collect()]


all_files = discover_files(RAW_DATA_PATH)
csv_files = [p for p in all_files if p.lower().endswith(".csv")]

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DATA_PATH}")

print("CSV files to ingest:")
for f in csv_files:
    print(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and clean CSV data

# COMMAND ----------

# DBTITLE 1,Cell 6
raw_csv_df = (
    spark.read.option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .csv(csv_files)
)

if len(raw_csv_df.columns) == 0:
    raise ValueError("CSV read succeeded but no columns were detected.")

# Create a canonical text source by concatenating all columns as key:value pairs.
kv_exprs = [
    F.concat(F.lit(f"{c}: "), F.coalesce(F.col(c).cast("string"), F.lit("")))
    for c in raw_csv_df.columns
]

csv_text_df = (
    raw_csv_df.withColumn("source_file", F.col("_metadata.file_path"))
    .withColumn("combined_text", F.concat_ws(" | ", *kv_exprs))
    .withColumn("combined_text", F.regexp_replace(F.col("combined_text"), r"\s+", " "))
    .withColumn("text_chunk_array", chunker_udf(F.col("combined_text")))
    .withColumn("source_type", F.lit("csv"))
)

exploded_df = (
    csv_text_df.withColumn("text_chunk", F.explode_outer(F.col("text_chunk_array")))
    .filter(F.col("text_chunk").isNotNull() & (F.length(F.col("text_chunk")) > 0))
)

window_spec = Window.orderBy(F.monotonically_increasing_id())
csv_chunks_df = (
    exploded_df.withColumn("chunk_id", F.row_number().over(window_spec))
    .withColumn(
        "metadata",
        F.to_json(
            F.struct(
                F.col("source_file").alias("file_name"),
                F.lit("csv_row").alias("record_type")
            )
        )
    )
    .select(
        F.col("chunk_id").cast("string").alias("id"),
        "text_chunk",
        "source_type",
        "metadata"
    )
)

if csv_chunks_df.count() == 0:
    raise ValueError("No non-empty text chunks were generated from CSV data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save CSV staging Delta table

# COMMAND ----------

(
    csv_chunks_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CSV_STAGING_TABLE)
)

print(f"Saved CSV staging table: {CSV_STAGING_TABLE}")
display(csv_chunks_df.limit(10))

# COMMAND ----------


