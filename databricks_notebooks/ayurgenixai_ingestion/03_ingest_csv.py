# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Ingest CSV
# MAGIC This notebook ingests CSV files from Unity Catalog volume storage, applies basic cleanup, creates RAG chunks, and stores a Delta staging table.

# COMMAND ----------

from pyspark.sql import functions as F
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

def _looks_like_missing_path(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(
        n in text
        for n in (
            "CloudFileNotFoundException",
            "Path does not exist",
            "No such file or directory",
            "FileNotFoundException",
        )
    )


def discover_files(path: str):
    placeholder_suffix = "/.placeholder"
    try:
        listed = dbutils.fs.ls(path)
        return [
            f.path for f in listed
            if not f.isDir() and not f.path.endswith(placeholder_suffix)
        ]
    except Exception as exc:
        if _looks_like_missing_path(exc):
            raise FileNotFoundError(
                f"Raw data directory does not exist: {path}\n"
                "Run notebook 01_setup_catalog_and_volume.py and upload the "
                "raw CSV/PDF files into that path before running this notebook."
            ) from exc
        print(f"dbutils.fs.ls failed for {path}. Falling back to Spark listing. Error: {exc}")
        files_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .load(path)
            .select(F.col("path"))
            .distinct()
        )
        return [
            r.path for r in files_df.collect()
            if not r.path.endswith(placeholder_suffix)
        ]


all_files = discover_files(RAW_DATA_PATH)
csv_files = [p for p in all_files if p.lower().endswith(".csv")]

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DATA_PATH}. Upload at least one .csv "
        "file (Catalog Explorer ▸ Volumes ▸ files ▸ raw_data ▸ Upload)."
    )

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
    csv_text_df.select(
        "*",
        F.posexplode_outer(F.col("text_chunk_array")).alias(
            "chunk_index_in_row", "text_chunk"
        ),
    )
    .filter(F.col("text_chunk").isNotNull() & (F.length(F.col("text_chunk")) > 0))
)

# Deterministic, parallel-friendly chunk_id. Avoids `row_number()` over a
# global (un-partitioned) window, which forces all rows into a single
# partition and triggers the Spark "No Partition Defined for Window" warning.
csv_chunks_df = (
    exploded_df.withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("source_file"), F.lit("unknown_file")),
                F.col("chunk_index_in_row").cast("string"),
                F.col("text_chunk"),
            ),
            256,
        ),
    )
    .withColumn(
        "metadata",
        F.to_json(
            F.struct(
                F.col("source_file").alias("file_name"),
                F.lit("csv_row").alias("record_type"),
                F.col("chunk_index_in_row").alias("chunk_index"),
            )
        ),
    )
    .select(
        F.col("chunk_id").alias("id"),
        "text_chunk",
        "source_type",
        "metadata",
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

