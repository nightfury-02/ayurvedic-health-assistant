# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Setup Catalog, Schema, and Volume
# MAGIC This notebook creates Unity Catalog objects for the AyurGenixAI ingestion pipeline.
# MAGIC
# MAGIC **Objects**
# MAGIC - Catalog: `bricksiitm`
# MAGIC - Schema: `ayurgenix`
# MAGIC - Volume: `files`

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"
VOLUME = "files"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_DATA_PATH = f"{VOLUME_ROOT}/raw_data/"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

print(f"Catalog ready: {CATALOG}")
print(f"Schema  ready: {CATALOG}.{SCHEMA}")
print(f"Volume  ready: {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Volume root  : {VOLUME_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify volume path
# MAGIC Expected raw data directory:
# MAGIC `/Volumes/bricksiitm/ayurgenix/files/raw_data/`

# COMMAND ----------

def ensure_raw_dir(path: str) -> None:
    try:
        dbutils.fs.ls(path)
        print(f"Raw data directory already exists: {path}")
        return
    except Exception:
        pass

    marker = path.rstrip("/") + "/.placeholder"
    try:
        dbutils.fs.put(marker, "placeholder", overwrite=True)
        print(f"Created raw data directory: {path}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create raw data directory {path}. "
            "Verify that you have WRITE VOLUME on "
            f"{CATALOG}.{SCHEMA}.{VOLUME}."
        ) from exc


ensure_raw_dir(RAW_DATA_PATH)
print(f"Raw data path configured: {RAW_DATA_PATH}")

# COMMAND ----------

import os


def _current_notebook_path() -> str:
    """Return the workspace path of the currently running notebook."""
    ctx = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
    )
    return ctx.notebookPath().get()


def _candidate_repo_raw_data_dirs(notebook_path: str):
    """Yield plausible workspace paths to the repo's `raw_data/` directory.

    Layout assumption (matches this repo):
        <repo_root>/databricks_notebooks/ayurgenixai_ingestion/01_setup_catalog_and_volume.py
        <repo_root>/raw_data/
    """
    workspace_root = "/Workspace" + notebook_path
    parts = workspace_root.split("/")
    for up in range(2, 6):
        if len(parts) <= up:
            continue
        candidate_root = "/".join(parts[:-up])
        yield f"{candidate_root}/raw_data"


def _list_local_files(local_dir: str):
    if not os.path.isdir(local_dir):
        return []
    out = []
    for name in os.listdir(local_dir):
        full = os.path.join(local_dir, name)
        if os.path.isfile(full) and (name.lower().endswith(".csv") or name.lower().endswith(".pdf")):
            out.append((name, full))
    return out


def _volume_is_empty(path: str) -> bool:
    listing = dbutils.fs.ls(path)
    real = [f for f in listing if not f.path.endswith("/.placeholder")]
    return len(real) == 0


def auto_upload_repo_raw_data(volume_path: str) -> int:
    """Copy CSV/PDF files from the repo's `raw_data/` folder into the volume.

    Returns the number of files uploaded. No-op if the volume already has
    real files, or if the repo folder cannot be located.
    """
    if not _volume_is_empty(volume_path):
        print("Volume already contains files; skipping auto-upload.")
        return 0

    notebook_path = _current_notebook_path()
    print(f"Notebook path: {notebook_path}")

    for candidate in _candidate_repo_raw_data_dirs(notebook_path):
        local_files = _list_local_files(candidate)
        if local_files:
            print(f"Found {len(local_files)} candidate file(s) in {candidate}")
            uploaded = 0
            for name, full in local_files:
                src = "file:" + full
                dst = volume_path.rstrip("/") + "/" + name
                dbutils.fs.cp(src, dst)
                print(f"  uploaded {name}")
                uploaded += 1
            return uploaded

    print(
        "Could not locate the repo's raw_data/ directory next to this notebook. "
        "Falling back to the manual upload instructions in the next cell."
    )
    return 0


uploaded_count = auto_upload_repo_raw_data(RAW_DATA_PATH)
print(f"Auto-uploaded files: {uploaded_count}")

# COMMAND ----------

try:
    listing = dbutils.fs.ls(RAW_DATA_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Cannot list {RAW_DATA_PATH}. The directory should exist after the "
        "previous cell. Re-run from the top, or check Unity Catalog volume "
        "permissions."
    ) from exc

real_files = [f for f in listing if not f.path.endswith("/.placeholder")]
print(f"Files currently in {RAW_DATA_PATH}: {len(real_files)}")
for f in real_files:
    print(f" - {f.path} ({f.size} bytes)")

if not real_files:
    print()
    print("⚠️  No data files found yet. Auto-upload could not locate the repo")
    print("   folder, so upload manually before running notebooks 02-07:")
    print()
    print(f"   a) UI: Catalog Explorer ▸ {CATALOG} ▸ {SCHEMA} ▸ Volumes ▸ {VOLUME}")
    print(f"      ▸ raw_data ▸ Upload (drag the files from raw_data/ in the repo).")
    print()
    print("   b) Databricks CLI (from your laptop, in the repo root):")
    print(f"      databricks fs cp -r raw_data {RAW_DATA_PATH}")
    print()
    print("   c) From a workspace cell, point at the workspace path of the repo:")
    print("      for src in dbutils.fs.ls('file:/Workspace/Users/<you>/<repo>/raw_data'):")
    print(f"          dbutils.fs.cp(src.path, '{RAW_DATA_PATH}' + src.name)")