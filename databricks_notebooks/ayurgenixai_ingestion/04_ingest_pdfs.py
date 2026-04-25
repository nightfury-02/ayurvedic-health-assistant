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


from io import BytesIO
from typing import List, Dict
from pyspark.sql import functions as F
import importlib.util
import os

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"
PDF_STAGING_TABLE = f"{CATALOG}.{SCHEMA}.pdf_chunks_staging"

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


def read_pdf_bytes(pdf_path: str) -> bytes:
    """
    Read PDF content from a UC Volume path without using local filesystem copies.
    """
    row = (
        spark.read.format("binaryFile")
        .load(pdf_path)
        .select("content")
        .first()
    )
    if row is None or row["content"] is None:
        raise ValueError(f"No binary content returned for {pdf_path}")
    return bytes(row["content"])


def extract_pdf_pages_with_fitz(pdf_bytes: bytes, file_name: str) -> List[Dict]:
    import fitz

    records = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
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


def extract_pdf_pages_with_pdfplumber(pdf_bytes: bytes, file_name: str) -> List[Dict]:
    import pdfplumber

    records = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
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
                "raw PDF files into that path before running this notebook."
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discover PDF files

# COMMAND ----------

all_files = discover_files(RAW_DATA_PATH)
pdf_files = [p for p in all_files if p.lower().endswith(".pdf")]

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in {RAW_DATA_PATH}. Upload at least one .pdf "
        "file (Catalog Explorer ▸ Volumes ▸ files ▸ raw_data ▸ Upload)."
    )

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
    try:
        pdf_bytes = read_pdf_bytes(pdf_path)
    except Exception as exc:
        failed_files.append((pdf_path, f"read_failed: {str(exc)}"))
        continue

    page_records = []
    try:
        page_records = extract_pdf_pages_with_fitz(pdf_bytes, file_name)
    except Exception:
        try:
            page_records = extract_pdf_pages_with_pdfplumber(pdf_bytes, file_name)
        except Exception as exc:
            failed_files.append((pdf_path, f"extract_failed: {str(exc)}"))
            page_records = []

    if not page_records:
        failed_files.append((pdf_path, "empty_or_unreadable_pdf"))
        continue

    valid_pages = [
        r for r in page_records
        if r.get("page_text") and str(r.get("page_text")).strip()
    ]
    if not valid_pages:
        failed_files.append((pdf_path, "no_non_empty_page_text"))
        continue

    full_text = "\n\n".join(
        str(r["page_text"]).strip() for r in sorted(valid_pages, key=lambda x: x["page_number"])
    )
    extracted_rows.append(
        {
            "file_name": file_name,
            "page_count": len(valid_pages),
            "document_text": full_text,
        }
    )

if failed_files:
    print("Some PDFs were skipped due to errors:")
    for item in failed_files:
        print(item)

if not extracted_rows:
    raise ValueError("No readable PDF content extracted from provided files.")

pdf_docs_df = spark.createDataFrame(extracted_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk text and prepare standardized schema

# COMMAND ----------


chunker_udf = F.udf(text_chunker, "array<string>")

chunked_df = (
    pdf_docs_df.withColumn("clean_text", F.regexp_replace(F.col("document_text"), r"\s+", " "))
    .withColumn("text_chunk_array", chunker_udf(F.col("clean_text")))
    .select(
        "*",
        F.posexplode_outer(F.col("text_chunk_array")).alias(
            "chunk_index_in_doc", "text_chunk"
        ),
    )
    .filter(F.col("text_chunk").isNotNull() & (F.length(F.col("text_chunk")) > 0))
    .withColumn("source_type", F.lit("pdf"))
)

# Deterministic, parallel-friendly chunk_id. Avoids `row_number()` over a
# global (un-partitioned) window, which forces all rows into a single
# partition and triggers the Spark "No Partition Defined for Window" warning.
pdf_chunks_df = (
    chunked_df.withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("file_name"), F.lit("unknown_file")),
                F.col("chunk_index_in_doc").cast("string"),
                F.col("text_chunk"),
            ),
            256,
        ),
    )
    .withColumn(
        "metadata",
        F.to_json(
            F.struct(
                F.col("file_name"),
                F.col("page_count"),
                F.lit("pdf_document").alias("record_type"),
                F.col("chunk_index_in_doc").alias("chunk_index"),
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



# COMMAND ----------

print(f"Total rows in pdf_chunks_df: {pdf_chunks_df.count()}")