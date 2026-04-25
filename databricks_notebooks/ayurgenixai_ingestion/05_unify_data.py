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

# MAGIC %pip install pymupdf pdfplumber
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()


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

# PDF text strategy:
#   "raw"       -> cleaned page text only (recommended for prose books).
#   "structured"-> force a herb-monograph template (only useful when most pages
#                  really are monographs; otherwise produces noisy
#                  "Not specified" filler that pollutes retrieval).
PDF_TEXT_STRATEGY = "raw"


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.split())


def clean_pdf_text(text: str) -> str:
    """
    Clean messy PDF text for embedding quality:
    - remove URLs
    - remove citation patterns: (1), [1], [1,2]
    - remove numbered references like "1)"
    - remove non-ASCII text (English-dominant fallback)
    - remove repeated symbols/artifacts
    - normalize whitespace
    """
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\d{1,4}\s*\)", " ", cleaned)
    cleaned = re.sub(r"\[\s*\d{1,4}(?:\s*,\s*\d{1,4})*\s*\]", " ", cleaned)
    cleaned = re.sub(r"(?:(?<=\s)|^)\d+\)\s*", " ", cleaned)
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)
    cleaned = re.sub(r"([^\w\s])\1{2,}", r"\1", cleaned)
    cleaned = re.sub(r"[_\-=:]{3,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_useless_line(line: str) -> bool:
    lowered = line.lower().strip()
    if not lowered:
        return True

    noisy_keywords = [
        "references",
        "bibliography",
        "journal",
        "doi",
        "http",
        "www",
        "pg.no",
        "edition",
        "vol.",
        "volume",
        "issue",
        "citation",
        "issn",
    ]
    if any(k in lowered for k in noisy_keywords):
        return True

    # Lines that are pure citation/page-number artifacts (digits + punctuation
    # only, no letters). Letter-containing lines like "1) Take 5g" are kept.
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
    filtered = clean_pdf_text(" ".join(useful_lines))
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


def build_pdf_raw_text(text: str, source_file: str = "", page_number: int = None) -> str:
    """Return cleaned PDF page text with a short, factual source header.

    This intentionally does NOT try to coerce arbitrary book pages into a
    herb-monograph template. That coercion (see :func:`structure_text`) is
    only meaningful for true monograph pages and otherwise injects
    "Not specified" filler into the retrieval index.
    """
    cleaned = clean_pdf_text(text)
    if not cleaned:
        return ""
    header = (
        f"Source File: {source_file or 'unknown_file'} | "
        f"Page Number: {page_number if page_number is not None else 'unknown'}"
    )
    return clean_text(f"{header} | {cleaned}")


def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    if text is None:
        return []

    normalized = clean_text(text)
    if not normalized:
        return []

    words = normalized.split()
    if len(words) <= chunk_size_words:
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


chunker_udf = F.udf(chunk_text, T.ArrayType(T.StringType()))
clean_text_udf = F.udf(clean_text, T.StringType())
clean_pdf_text_udf = F.udf(clean_pdf_text, T.StringType())
structure_text_udf = F.udf(structure_text, T.StringType())
build_pdf_raw_text_udf = F.udf(build_pdf_raw_text, T.StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discover input files

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


def discover_files(path: str) -> List[str]:
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
csv_files = [f for f in all_files if f.lower().endswith(".csv")]
pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

if not csv_files and not pdf_files:
    raise FileNotFoundError(
        f"No CSV or PDF files found in {RAW_DATA_PATH}. The directory exists "
        "but is empty. Upload the source files (Catalog Explorer ▸ Volumes ▸ "
        "files ▸ raw_data ▸ Upload) and re-run."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Structured text creation for CSV rows

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


def discover_files(path: str) -> List[str]:
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
csv_files = [f for f in all_files if f.lower().endswith(".csv")]
pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

if not csv_files and not pdf_files:
    raise FileNotFoundError(
        f"No CSV or PDF files found in {RAW_DATA_PATH}. The directory exists "
        "but is empty. Upload the source files (Catalog Explorer ▸ Volumes ▸ "
        "files ▸ raw_data ▸ Upload) and re-run."
    )

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


# Pre-flight: confirm at least one PDF backend is importable. Otherwise we
# would skip every single PDF with the same "No module named ..." message.
_pdf_backends_available = []
try:
    import fitz  # noqa: F401  (PyMuPDF)
    _pdf_backends_available.append("pymupdf")
except Exception as _fitz_exc:
    _fitz_import_error = _fitz_exc

try:
    import pdfplumber  # noqa: F401
    _pdf_backends_available.append("pdfplumber")
except Exception as _plumber_exc:
    _plumber_import_error = _plumber_exc

if pdf_files and not _pdf_backends_available:
    raise ImportError(
        "Neither pymupdf nor pdfplumber is installed on this cluster, so no "
        "PDF can be parsed. Run the install cell at the top of this notebook "
        "(`%pip install pymupdf pdfplumber` followed by "
        "`dbutils.library.restartPython()`), then re-run from this cell."
    )

print(f"PDF backends available: {_pdf_backends_available}")

pdf_rows = []
pdf_failures = []
for pdf_path in pdf_files:
    file_name = pdf_path.split("/")[-1]
    try:
        pdf_bytes = read_pdf_bytes(pdf_path)
    except Exception as exc:
        pdf_failures.append((pdf_path, f"read_failed: {exc}"))
        continue

    extraction_errors = []
    records = None

    if "pymupdf" in _pdf_backends_available:
        try:
            records = extract_pdf_pages_with_fitz(pdf_bytes, file_name)
        except Exception as exc:
            extraction_errors.append(f"pymupdf: {exc}")

    if not records and "pdfplumber" in _pdf_backends_available:
        try:
            records = extract_pdf_pages_with_pdfplumber(pdf_bytes, file_name)
        except Exception as exc:
            extraction_errors.append(f"pdfplumber: {exc}")

    if records:
        pdf_rows.extend(records)
    else:
        pdf_failures.append((pdf_path, "; ".join(extraction_errors) or "no extractor produced text"))

if pdf_failures:
    print(f"Skipped {len(pdf_failures)} PDF(s):")
    for path, reason in pdf_failures:
        print(f"  - {path}: {reason}")

if pdf_rows:
    pdf_pages_df = spark.createDataFrame(pdf_rows)

    if PDF_TEXT_STRATEGY == "structured":
        raw_text_expr = structure_text_udf(
            F.col("page_text"),
            F.col("source_file"),
            F.col("page_number"),
        )
    else:
        raw_text_expr = build_pdf_raw_text_udf(
            F.col("page_text"),
            F.col("source_file"),
            F.col("page_number"),
        )

    pdf_structured_df = (
        pdf_pages_df.withColumn("source_type", F.lit("pdf"))
        .withColumn("page_text", clean_pdf_text_udf(F.col("page_text")))
        .filter(F.length(F.trim(F.col("page_text"))) > 0)
        .withColumn("raw_text", raw_text_expr)
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
    .withColumn("alpha_chars", F.length(F.regexp_replace(F.col("chunk_text"), r"[^A-Za-z]", "")))
    .withColumn("visible_chars", F.length(F.regexp_replace(F.col("chunk_text"), r"\s+", "")))
    .withColumn(
        "alpha_ratio",
        F.when(F.col("visible_chars") > 0, F.col("alpha_chars") / F.col("visible_chars")).otherwise(F.lit(0.0)),
    )
    .filter(F.col("chunk_word_count") >= 10)
    .filter(F.col("alpha_ratio") >= 0.2)
    .filter(~F.lower(F.col("chunk_text")).rlike(r"^\s*(references|bibliography|doi|journal|www|http)\s*$"))
    .drop("chunk_word_count", "alpha_chars", "visible_chars", "alpha_ratio")
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

