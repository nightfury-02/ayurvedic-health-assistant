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
import re
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


def clean_pdf_text(text: str) -> str:
    """
    Light-touch cleaning to preserve semantic content:
    - remove URLs
    - soften citation noise
    - normalize repeated punctuation artifacts
    - normalize whitespace
    """
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\d{1,4}\s*\)", " ", cleaned)
    cleaned = re.sub(r"\[\s*\d{1,4}(?:\s*,\s*\d{1,4})*\s*\]", " ", cleaned)
    cleaned = re.sub(r"(?:(?<=\s)|^)\d+\)\s*", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"([^\w\s])\1{2,}", r"\1", cleaned)
    cleaned = re.sub(r"[_\-=:]{4,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_useless_line(line: str) -> bool:
    lowered = line.lower().strip()
    if not lowered:
        return True

    # Keep this intentionally conservative: remove only clear metadata/noise rows.
    if lowered in {"references", "bibliography"}:
        return True
    if lowered.startswith("doi:"):
        return True
    if "http" in lowered or "www" in lowered:
        return True

    # Lines that are mostly citation artifacts.
    if re.fullmatch(r"[\[\]\(\)\d,\.\s\-:;]+", lowered):
        return True

    return False


def _extract_field(text: str, labels: List[str]) -> str:
    for label in labels:
        pattern = rf"(?i)\b{label}\b\s*[:\-]\s*(.{5,220}?)(?=(?:\s+\b[A-Za-z ]{{2,35}}\s*[:\-])|$)"
        match = re.search(pattern, text)
        if match:
            return clean_text(match.group(1))
    return ""


def structure_text(text: str, source_file: str = "", page_number: int = None) -> str:
    """
    Convert noisy PDF text into semi-structured, labeled blocks.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    useful_lines = [ln for ln in lines if not _is_useless_line(ln)]
    # If conservative line filtering still collapses content, fall back to all lines.
    filtered = clean_pdf_text(" ".join(useful_lines) if useful_lines else " ".join(lines))
    if not filtered:
        return ""

    ingredient = _extract_field(filtered, ["ingredient", "herb", "drug", "name"])
    botanical = _extract_field(filtered, ["botanical name", "latin name", "scientific name"])
    characteristics = _extract_field(filtered, ["characteristics", "properties", "description"])
    uses = _extract_field(filtered, ["uses", "benefits", "indications"])
    kitchen_use = _extract_field(filtered, ["kitchen use", "culinary use", "food use"])
    ayurvedic_use = _extract_field(filtered, ["ayurvedic use", "ayurvedic uses", "ayurveda"])
    remedies = _extract_field(filtered, ["remedies", "home remedy", "preparation", "dosage"])

    # Fallback extraction for pages without explicit labels.
    if not ingredient:
        candidate = filtered.split(".")[0]
        ingredient = " ".join(candidate.split()[:8]).strip()

    if not uses:
        uses = " ".join(filtered.split()[:40]).strip()

    extraction_hit_count = sum(
        1 for value in [ingredient, botanical, characteristics, uses, kitchen_use, ayurvedic_use, remedies] if value
    )
    if extraction_hit_count <= 2:
        # Regex extraction is fragile for free-form PDFs; do not collapse to synthetic fields.
        return clean_text(
            f"Source File: {source_file or 'unknown_file'} | "
            f"Page Number: {str(page_number) if page_number is not None else 'unknown'} | "
            f"Content: {filtered}"
        )

    parts = [
        f"Ingredient: {ingredient}" if ingredient else "Ingredient: Unknown",
        f"Botanical Name: {botanical}" if botanical else "Botanical Name: Not specified",
        f"Characteristics: {characteristics}" if characteristics else "Characteristics: Not specified",
        f"Uses: {uses}" if uses else "Uses: Not specified",
        f"Kitchen Use: {kitchen_use}" if kitchen_use else "Kitchen Use: Not specified",
        f"Ayurvedic Use: {ayurvedic_use}" if ayurvedic_use else "Ayurvedic Use: Not specified",
        f"Remedies: {remedies}" if remedies else "Remedies: Not specified",
        f"Source File: {source_file or 'unknown_file'}",
        f"Page Number: {str(page_number) if page_number is not None else 'unknown'}",
    ]

    return clean_text(" | ".join(parts))


def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    if text is None:
        return []

    normalized = clean_text(text)
    if not normalized:
        return []

    words = normalized.split()
    if len(words) <= 400:
        # Required behavior: do not split small text.
        return [normalized]

    chunks: List[str] = []
    step = max(chunk_size_words - overlap_words, 1)
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step
    return chunks


def translate_to_english_placeholder(text: str) -> str:
    """
    Placeholder for future IndicTrans2 integration.
    Keep pass-through for now to avoid runtime dependency.
    """
    return text


def sanitize_column_name(column_name: str) -> str:
    return re.sub(r"\s+", "_", str(column_name or "").strip().lower())


chunker_udf = F.udf(chunk_text, T.ArrayType(T.StringType()))
clean_text_udf = F.udf(clean_text, T.StringType())
clean_pdf_text_udf = F.udf(clean_pdf_text, T.StringType())
structure_text_udf = F.udf(structure_text, T.StringType())

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
# MAGIC ## Structured text creation for CSV rows (dynamic all-columns)

# COMMAND ----------

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

    metadata_like_cols = {"source_file", "_metadata"}
    csv_content_cols = [
        c for c in csv_base_df.columns if sanitize_column_name(c) not in metadata_like_cols and c != "source_file"
    ]
    if not csv_content_cols:
        raise ValueError("CSV files detected but no content columns are available for ingestion.")

    print(f"[DEBUG][CSV] Detected columns ({len(csv_content_cols)}): {csv_content_cols}")
    rich_parts = [
        F.when(
            F.trim(clean_text_udf(F.col(c).cast("string"))) != "",
            F.concat(F.lit(f"{c}: "), clean_text_udf(F.col(c).cast("string"))),
        )
        for c in csv_content_cols
    ]
    summary_parts = [clean_text_udf(F.col(c).cast("string")) for c in csv_content_cols]

    csv_structured_df = (
        csv_base_df.withColumn("source_type", F.lit("csv"))
        .withColumn("page_number", F.lit(None).cast("int"))
        .withColumn(
            "parts",
            F.array(
                F.lit("Source: csv"),
                F.concat(F.lit("Case Summary: "), F.concat_ws(". ", *summary_parts)),
                *rich_parts,
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

print(f"[DEBUG] CSV files discovered: {len(csv_files)}")
if csv_files:
    print("[DEBUG] Sample CSV raw_text:")
    display(csv_structured_df.select("raw_text", "source_file").limit(3))

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
    print(f"[DEBUG][PDF] Extracted page rows: {pdf_pages_df.count()}")
    print("[DEBUG][PDF] Sample extracted page_text (raw):")
    display(pdf_pages_df.select("source_file", "page_number", "page_text").limit(2))
    pdf_structured_df = (
        pdf_pages_df.withColumn("source_type", F.lit("pdf"))
        .withColumn("clean_page_text", clean_pdf_text_udf(F.col("page_text")))
        .filter(F.length(F.trim(F.col("clean_page_text"))) > 0)
        .withColumn(
            "structured_text",
            structure_text_udf(
                F.col("clean_page_text"),
                F.col("source_file"),
                F.col("page_number"),
            ),
        )
        .withColumn(
            "raw_text",
            F.when(
                F.length(F.trim(F.col("structured_text"))) > 0,
                F.concat(
                    F.col("structured_text"),
                    F.lit(" | Full Page Text: "),
                    F.col("clean_page_text"),
                ),
            ).otherwise(F.col("clean_page_text")),
        )
        .withColumn(
            "raw_text",
            F.concat(
                F.lit("Source: pdf | Source File: "),
                F.coalesce(F.col("source_file"), F.lit("unknown_file")),
                F.lit(" | Page Number: "),
                F.coalesce(F.col("page_number").cast("string"), F.lit("unknown")),
                F.lit(" | "),
                F.col("raw_text"),
            ),
        )        
        .filter(F.length(F.trim(F.col("raw_text"))) > 0)
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

# Unified pipeline: include both CSV and PDF records.
base_df = csv_structured_df.unionByName(pdf_structured_df, allowMissingColumns=True)
print(f"[DEBUG] Structured row count (CSV + PDF): {base_df.count()}")
if base_df.count() == 0:
    raise ValueError("No structured records built from CSV/PDF inputs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean text and chunk with boundary preservation

# COMMAND ----------

# DBTITLE 1,Cell 10
chunked_df = (
    base_df.withColumn("raw_text", clean_text_udf(F.col("raw_text")))
    .filter(F.length(F.trim(F.col("raw_text"))) > 0)
    .withColumn("chunk_array", chunker_udf(F.col("raw_text")))
    .select("*", F.posexplode_outer(F.col("chunk_array")).alias("chunk_index", "chunk_text"))
    .select(
        "row_id",
        "chunk_index",
        "chunk_text",
        "source_type",
        "source_file",
        "page_number",
    )
    .filter(F.col("chunk_text").isNotNull() & (F.length(F.trim(F.col("chunk_text"))) > 0))
)

filtered_chunked_df = (
    chunked_df.withColumn("chunk_word_count", F.size(F.split(F.trim(F.col("chunk_text")), r"\s+")))
    .filter(F.col("chunk_word_count") >= 5)
    .filter(~F.lower(F.col("chunk_text")).rlike(r"^\s*(references|bibliography|doi|journal|www|http)\s*$"))
    .drop("chunk_word_count")
)

final_df = (
    filtered_chunked_df.withColumn("chunk_index", F.col("chunk_index").cast("int"))
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

print(f"[DEBUG] Chunked rows before filtering: {chunked_df.count()}")
print(f"[DEBUG] Final chunk count after filtering: {final_df.count()}")
print("[DEBUG] Sample final chunks:")
display(final_df.select("chunk_text", "source_type", "source_file", "page_number").limit(5))

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


