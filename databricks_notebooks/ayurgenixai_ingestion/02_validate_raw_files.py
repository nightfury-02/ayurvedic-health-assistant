# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Validate Raw Files
# MAGIC This notebook lists all files under the raw Unity Catalog volume path and separates CSV and PDF inputs.

# COMMAND ----------

from pyspark.sql import functions as F

RAW_DATA_PATH = "/Volumes/bricksiitm/ayurgenix/files/raw_data/"


class RawDataMissingError(RuntimeError):
    """Raised when the raw-data directory does not exist or is empty."""


def _looks_like_missing_path(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    needles = (
        "CloudFileNotFoundException",
        "Path does not exist",
        "No such file or directory",
        "FileNotFoundException",
    )
    return any(n in text for n in needles)


def discover_files(path: str):
    """
    Discover files from a Unity Catalog volume path.

    - Tries dbutils.fs.ls first.
    - Falls back to Spark binaryFile listing for permission edge cases.
    - Raises a clear, actionable RawDataMissingError when the directory does
      not exist (no JVM stacktrace).
    """
    placeholder_suffix = "/.placeholder"

    try:
        listed = dbutils.fs.ls(path)
        return [
            f.path for f in listed
            if not f.isDir() and not f.path.endswith(placeholder_suffix)
        ]
    except Exception as exc:
        if _looks_like_missing_path(exc):
            raise RawDataMissingError(
                f"Raw data directory does not exist: {path}\n"
                "Run notebook 01_setup_catalog_and_volume.py first, then upload "
                "the CSV/PDF files into that path (Catalog Explorer ▸ Volumes ▸ "
                "files ▸ raw_data ▸ Upload, or `databricks fs cp -r raw_data "
                f"{path}`)."
            ) from exc

        print(
            f"dbutils.fs.ls failed for {path}. Falling back to Spark listing. "
            f"Error: {exc}"
        )
        try:
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
        except Exception as fallback_exc:
            if _looks_like_missing_path(fallback_exc):
                raise RawDataMissingError(
                    f"Raw data directory does not exist: {path}\n"
                    "Run notebook 01 first and upload the raw files into "
                    f"{path}."
                ) from fallback_exc
            raise RuntimeError(f"Failed to list files in {path}") from fallback_exc


all_paths = discover_files(RAW_DATA_PATH)
if not all_paths:
    raise RawDataMissingError(
        f"No files found in {RAW_DATA_PATH}.\n"
        "The directory exists but is empty. Upload raw CSV/PDF files into it "
        "(Catalog Explorer ▸ Volumes ▸ files ▸ raw_data ▸ Upload), then re-run."
    )

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

# COMMAND ----------

