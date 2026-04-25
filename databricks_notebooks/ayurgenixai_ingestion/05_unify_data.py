# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Build Processed Knowledge Base
# MAGIC This notebook reads raw CSV/PDF files and applies robust preprocessing:
# MAGIC - context-aware chunking on `raw_text`
# MAGIC - source metadata preservation
# MAGIC - deterministic chunk metadata
# MAGIC
# MAGIC Final output table:
# MAGIC `bricksiitm.ayurgenix.processed_knowledge_base`

# COMMAND ----------

from io import BytesIO
from typing import Dict, List
from pyspark.sql import functions as F
from pyspark.sql import types as T

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

PROCESSED_TABLE = f"{CATALOG}.{SCHEMA}.processed_knowledge_base"
RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"

# Chunking requirements:
# - 300-400 words per chunk
# - 80-100 word overlap
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 90


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.split())


def chunk_text_by_words(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    if text is None:
        return []

    normalized = clean_text(text)
    if not normalized:
        return []

    words = normalized.split()
    if len(words) < chunk_size:
        # Required behavior: do not split small text.
        return [normalized]

    segments = [clean_text(part) for part in normalized.split("|")]
    segments = [s for s in segments if s]
    if not segments:
        return [normalized]

    # First pass: keep logical structure by grouping whole "|" segments.
    base_chunks: List[str] = []
    current_segments: List[str] = []
    current_len = 0

    for segment in segments:
        seg_words = segment.split()
        seg_len = len(seg_words)

        if seg_len > chunk_size:
            # Flush current chunk before splitting very large segment.
            if current_segments:
                base_chunks.append(" | ".join(current_segments))
                current_segments = []
                current_len = 0

            start = 0
            while start < seg_len:
                end = min(start + chunk_size, seg_len)
                piece = " ".join(seg_words[start:end])
                base_chunks.append(piece)
                if end >= seg_len:
                    break
                start += max(chunk_size - overlap, 1)
            continue

        projected_len = current_len + seg_len
        if current_segments:
            projected_len += 1  # for visual "|" separator boundary

        if current_segments and projected_len > chunk_size:
            base_chunks.append(" | ".join(current_segments))
            current_segments = [segment]
            current_len = seg_len
        else:
            current_segments.append(segment)
            current_len = projected_len if current_len > 0 else seg_len

    if current_segments:
        base_chunks.append(" | ".join(current_segments))

    # Second pass: word overlap for retrieval continuity.
    overlapped_chunks: List[str] = []
    for idx, chunk in enumerate(base_chunks):
        if idx == 0:
            overlapped_chunks.append(chunk)
            continue

        prev_words = base_chunks[idx - 1].split()
        tail = prev_words[-overlap:] if len(prev_words) > overlap else prev_words
        candidate_words = tail + chunk.split()

        if len(candidate_words) > chunk_size:
            candidate_words = candidate_words[:chunk_size]

        overlapped_chunks.append(" ".join(candidate_words))

    return [c for c in overlapped_chunks if c]


chunker_udf = F.udf(chunk_text_by_words, T.ArrayType(T.StringType()))
clean_text_udf = F.udf(clean_text, T.StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discover input files

# COMMAND ----------

def discover_files(path: str) -> List[str]:
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
csv_files = [f for f in all_files if f.lower().endswith(".csv")]
pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

if not csv_files and not pdf_files:
    raise FileNotFoundError(f"No CSV or PDF files found in {RAW_DATA_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Structured text creation for CSV rows

# COMMAND ----------

def choose_column(candidates: List[str], available: List[str]) -> str:
    available_norm = {c.lower().strip() for c in available}
    for candidate in candidates:
        if candidate in available_norm:
            return candidate
    return ""


csv_base_df = None
if csv_files:
    csv_base_df = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(csv_files)
        .withColumn("source_file", F.col("_metadata.file_path"))
    )

    available_cols = [c.lower().strip() for c in csv_base_df.columns]
    col_lookup = {c.lower().strip(): c for c in csv_base_df.columns}

    condition_col = choose_column(["condition", "disease", "ailment", "problem"], available_cols)
    diet_col = choose_column(["diet", "diet_recommendation", "food", "ahara"], available_cols)
    yoga_col = choose_column(["yoga", "asana", "exercise", "lifestyle"], available_cols)
    medical_col = choose_column(["medical", "medicine", "treatment", "remedy"], available_cols)
    prevention_col = choose_column(["prevention", "preventive", "care", "tips"], available_cols)
    prognosis_col = choose_column(["prognosis", "outlook", "recovery"], available_cols)

    def col_or_empty(column_name: str):
        if not column_name:
            return F.lit(None).cast("string")
        return F.col(col_lookup[column_name]).cast("string")

    condition_expr = clean_text_udf(col_or_empty(condition_col))
    diet_expr = clean_text_udf(col_or_empty(diet_col))
    yoga_expr = clean_text_udf(col_or_empty(yoga_col))
    medical_expr = clean_text_udf(col_or_empty(medical_col))
    prevention_expr = clean_text_udf(col_or_empty(prevention_col))
    prognosis_expr = clean_text_udf(col_or_empty(prognosis_col))

    csv_structured_df = (
        csv_base_df.withColumn("source_type", F.lit("csv"))
        .withColumn("page_number", F.lit(None).cast("int"))
        .withColumn(
            "parts",
            F.array(
                F.lit("Source: csv"),
                F.when(F.trim(condition_expr) != "", F.concat(F.lit("Condition: "), condition_expr)),
                F.when(F.trim(diet_expr) != "", F.concat(F.lit("Diet: "), diet_expr)),
                F.when(F.trim(yoga_expr) != "", F.concat(F.lit("Yoga: "), yoga_expr)),
                F.when(F.trim(medical_expr) != "", F.concat(F.lit("Medical: "), medical_expr)),
                F.when(F.trim(prevention_expr) != "", F.concat(F.lit("Prevention: "), prevention_expr)),
                F.when(F.trim(prognosis_expr) != "", F.concat(F.lit("Prognosis: "), prognosis_expr)),
                F.concat(F.lit("Source File: "), F.coalesce(F.col("source_file"), F.lit("unknown_file"))),
            ),
        )
        .withColumn("parts", F.expr("filter(parts, x -> x is not null and trim(x) <> '')"))
        .withColumn("raw_text", F.concat_ws(" | ", F.col("parts")))
        .select(
            F.expr("uuid()").alias("row_id"),
            "raw_text",
            "source_type",
            "source_file",
            "page_number",
        )
    )
else:
    csv_structured_df = spark.createDataFrame(
        [],
        "row_id string, raw_text string, source_type string, source_file string, page_number int",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Structured text creation for PDF pages

# COMMAND ----------

def read_pdf_bytes(pdf_path: str) -> bytes:
    row = (
        spark.read.format("binaryFile")
        .load(pdf_path)
        .select("content")
        .first()
    )
    if row is None or row["content"] is None:
        raise ValueError(f"No binary content for {pdf_path}")
    return bytes(row["content"])


def extract_pdf_pages_with_fitz(pdf_bytes: bytes, file_name: str) -> List[Dict]:
    import fitz

    rows = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for idx, page in enumerate(doc, start=1):
            rows.append(
                {
                    "source_file": file_name,
                    "page_number": idx,
                    "page_text": page.get_text("text") or "",
                }
            )
    return rows


def extract_pdf_pages_with_pdfplumber(pdf_bytes: bytes, file_name: str) -> List[Dict]:
    import pdfplumber

    rows = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            rows.append(
                {
                    "source_file": file_name,
                    "page_number": idx,
                    "page_text": page.extract_text() or "",
                }
            )
    return rows


pdf_rows = []
for pdf_path in pdf_files:
    file_name = pdf_path.split("/")[-1]
    try:
        pdf_bytes = read_pdf_bytes(pdf_path)
        try:
            records = extract_pdf_pages_with_fitz(pdf_bytes, file_name)
        except Exception:
            records = extract_pdf_pages_with_pdfplumber(pdf_bytes, file_name)
        pdf_rows.extend(records)
    except Exception as exc:
        print(f"Skipping unreadable PDF: {pdf_path} ({str(exc)})")

if pdf_rows:
    pdf_pages_df = spark.createDataFrame(pdf_rows)
    pdf_structured_df = (
        pdf_pages_df.withColumn("source_type", F.lit("pdf"))
        .withColumn("page_text", clean_text_udf(F.col("page_text")))
        .withColumn(
            "raw_text",
            F.concat(
                F.lit("Source: pdf"),
                F.lit(" | Source File: "),
                F.col("source_file"),
                F.lit(" | Page Number: "),
                F.col("page_number").cast("string"),
                F.lit(" | Content: "),
                F.col("page_text"),
            ),
        )
        .select(
            F.expr("uuid()").alias("row_id"),
            "raw_text",
            "source_type",
            "source_file",
            "page_number",
        )
    )
else:
    pdf_structured_df = spark.createDataFrame(
        [],
        "row_id string, raw_text string, source_type string, source_file string, page_number int",
    )

base_df = csv_structured_df.unionByName(pdf_structured_df, allowMissingColumns=True)
if base_df.count() == 0:
    raise ValueError("No structured records built from CSV/PDF inputs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean text and chunk with boundary preservation

# COMMAND ----------

chunked_df = (
    base_df.withColumn("raw_text", clean_text_udf(F.col("raw_text")))
    .withColumn("chunk_array", chunker_udf(F.col("raw_text")))
    .withColumn("chunk_with_pos", F.posexplode_outer(F.col("chunk_array")))
    .select(
        "row_id",
        F.col("chunk_with_pos.pos").alias("chunk_index"),
        F.col("chunk_with_pos.col").alias("chunk_text"),
        "source_type",
        "source_file",
        "page_number",
    )
    .filter(F.col("chunk_text").isNotNull() & (F.length(F.trim(F.col("chunk_text"))) > 0))
)

final_df = (
    chunked_df.withColumn("chunk_index", F.col("chunk_index").cast("int"))
    .withColumn("chunk_text", clean_text_udf(F.col("chunk_text")))
    .withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("row_id"),
                F.col("chunk_index").cast("string"),
                F.col("chunk_text"),
            ),
            256,
        ),
    )
    .select(
        "row_id",
        "chunk_id",
        "chunk_index",
        "chunk_text",
        "source_type",
        "source_file",
        F.col("page_number").cast("int").alias("page_number"),
    )
)

if final_df.count() == 0:
    raise ValueError("No chunks produced after context-aware preprocessing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save final processed table

# COMMAND ----------

try:
    (
        final_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PROCESSED_TABLE)
    )
except Exception as exc:
    raise RuntimeError(f"Failed writing Delta table: {PROCESSED_TABLE}") from exc

print(f"Saved processed knowledge base: {PROCESSED_TABLE}")
display(final_df.limit(20))

# COMMAND ----------