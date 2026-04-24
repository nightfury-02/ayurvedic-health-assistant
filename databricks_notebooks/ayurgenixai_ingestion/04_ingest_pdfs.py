# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Ingest PDFs
# MAGIC This notebook extracts text from PDF files in the Unity Catalog volume path and writes chunked output to a Delta staging table.
# MAGIC
# MAGIC It attempts extraction with **PyMuPDF** first, then falls back to **pdfplumber** for compatibility.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional dependency install
# MAGIC Run once per cluster if these libraries are not preinstalled.

# COMMAND ----------

# MAGIC %pip install pymupdf pdfplumber

# COMMAND ----------

from typing import List, Dict
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"
PDF_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.pdf_chunks_staging"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def text_chunker(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if text is None:
        return []
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += step
    return chunks


def extract_pdf_pages_with_fitz(local_path: str, file_name: str) -> List[Dict]:
    import fitz

    records = []
    with fitz.open(local_path) as doc:
        for idx, page in enumerate(doc, start=1):
            txt = page.get_text("text") or ""
            records.append(
                {
                    "file_name": file_name,
                    "page_number": idx,
                    "page_text": txt
                }
            )
    return records


def extract_pdf_pages_with_pdfplumber(local_path: str, file_name: str) -> List[Dict]:
    import pdfplumber

    records = []
    with pdfplumber.open(local_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            records.append(
                {
                    "file_name": file_name,
                    "page_number": idx,
                    "page_text": txt
                }
            )
    return records


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discover PDF files

# COMMAND ----------

all_files = discover_files(RAW_DATA_PATH)
pdf_files = [p for p in all_files if p.lower().endswith(".pdf")]

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {RAW_DATA_PATH}")

print("PDF files to ingest:")
for p in pdf_files:
    print(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract text (driver-side) and build Spark DataFrame

# COMMAND ----------

extracted_rows = []
failed_files = []

for pdf_path in pdf_files:
    file_name = pdf_path.split("/")[-1]
    local_copy = f"/tmp/{file_name}"
    try:
        dbutils.fs.cp(pdf_path, f"file:{local_copy}", recurse=False)
    except Exception as exc:
        failed_files.append((pdf_path, f"copy_failed: {str(exc)}"))
        continue

    page_records = []
    try:
        page_records = extract_pdf_pages_with_fitz(local_copy, file_name)
    except Exception:
        try:
            page_records = extract_pdf_pages_with_pdfplumber(local_copy, file_name)
        except Exception as exc:
            failed_files.append((pdf_path, f"extract_failed: {str(exc)}"))
            page_records = []

    if not page_records:
        failed_files.append((pdf_path, "empty_or_unreadable_pdf"))
        continue

    extracted_rows.extend(page_records)

if failed_files:
    print("Some PDFs were skipped due to errors:")
    for item in failed_files:
        print(item)

if not extracted_rows:
    raise ValueError("No readable PDF content extracted from provided files.")

pdf_pages_df = spark.createDataFrame(extracted_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk text and prepare standardized schema

# COMMAND ----------

chunker_udf = F.udf(text_chunker, "array<string>")

chunked_df = (
    pdf_pages_df.withColumn("clean_text", F.regexp_replace(F.col("page_text"), r"\s+", " "))
    .withColumn("text_chunk_array", chunker_udf(F.col("clean_text")))
    .withColumn("text_chunk", F.explode_outer(F.col("text_chunk_array")))
    .filter(F.col("text_chunk").isNotNull() & (F.length(F.col("text_chunk")) > 0))
    .withColumn("source_type", F.lit("pdf"))
)

window_spec = Window.orderBy(F.monotonically_increasing_id())
pdf_chunks_df = (
    chunked_df.withColumn("chunk_id", F.row_number().over(window_spec))
    .withColumn(
        "metadata",
        F.to_json(
            F.struct(
                F.col("file_name"),
                F.col("page_number"),
                F.lit("pdf_page").alias("record_type")
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

if pdf_chunks_df.count() == 0:
    raise ValueError("PDF ingestion completed but no text chunks were produced.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save PDF staging Delta table

# COMMAND ----------

(
    pdf_chunks_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PDF_STAGING_TABLE)
)

print(f"Saved PDF staging table: {PDF_STAGING_TABLE}")
display(pdf_chunks_df.limit(10))
