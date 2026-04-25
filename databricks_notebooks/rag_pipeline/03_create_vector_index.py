# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Create Vector Search Endpoint and Index
# MAGIC
# MAGIC This notebook creates:
# MAGIC - Vector Search endpoint
# MAGIC - Delta Sync index on `bricksiitm.ayurgenix.knowledge_base_embeddings`
# MAGIC
# MAGIC Adjust names below once and re-run safely.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings"
VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"

# Match the embedding dimension of your embedding model.
EMBEDDING_DIMENSION = 1024
EMBEDDING_COLUMN = "embedding"
PRIMARY_KEY = "chunk_id"

print(f"CATALOG/SCHEMA       : {CATALOG}.{SCHEMA}")
print(f"EMBEDDINGS_TABLE     : {EMBEDDINGS_TABLE}")
print(f"VECTOR_ENDPOINT_NAME : {VECTOR_ENDPOINT_NAME}")
print(f"VECTOR_INDEX_NAME    : {VECTOR_INDEX_NAME}")

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed on source table
print(f"Enabling Change Data Feed on {EMBEDDINGS_TABLE}...")

spark.sql(
    f"""
    ALTER TABLE {EMBEDDINGS_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)

print("Change Data Feed enabled.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create endpoint and index

# COMMAND ----------

# DBTITLE 1,Cell 5


import time

from databricks.vector_search.client import VectorSearchClient

# Endpoint type:
#   "STANDARD"     - serverless, billed per use (recommended for most use cases)
#   "STORAGE_OPTIMIZED" - for very large indexes (>320M vectors)
ENDPOINT_TYPE = "STANDARD"

# Max wait when creating a brand-new endpoint (provisioning can take 5-15 min).
ENDPOINT_WAIT_SECONDS = 30 * 60
ENDPOINT_POLL_SECONDS = 30


def _endpoint_exists(client: VectorSearchClient, name: str) -> bool:
    """Return True if a Vector Search endpoint with this name exists."""
    try:
        endpoints = client.list_endpoints().get("endpoints", []) or []
    except Exception as exc:  # noqa: BLE001
        print(f"Could not list endpoints ({exc!s}); assuming endpoint missing.")
        return False
    return any(ep.get("name") == name for ep in endpoints)


def _endpoint_state(client: VectorSearchClient, name: str) -> str:
    info = client.get_endpoint(name) or {}
    return (info.get("endpoint_status") or {}).get("state", "UNKNOWN")


def ensure_endpoint(client: VectorSearchClient, name: str) -> None:
    """Create the endpoint if missing and block until it is ONLINE."""
    if not _endpoint_exists(client, name):
        print(f"Endpoint '{name}' not found. Creating ({ENDPOINT_TYPE})...")
        client.create_endpoint(name=name, endpoint_type=ENDPOINT_TYPE)
    else:
        print(f"Endpoint '{name}' already exists.")

    deadline = time.time() + ENDPOINT_WAIT_SECONDS
    last_state = None
    while time.time() < deadline:
        state = _endpoint_state(client, name)
        if state != last_state:
            print(f"  endpoint state: {state}")
            last_state = state
        if state == "ONLINE":
            return
        if state in {"FAILED", "OFFLINE"}:
            raise RuntimeError(f"Endpoint '{name}' is in unrecoverable state: {state}")
        time.sleep(ENDPOINT_POLL_SECONDS)
    raise TimeoutError(
        f"Endpoint '{name}' did not reach ONLINE within {ENDPOINT_WAIT_SECONDS}s "
        f"(last observed state: {last_state})."
    )


vsc = VectorSearchClient()
ensure_endpoint(vsc, VECTOR_ENDPOINT_NAME)
print(f"Endpoint ready: {VECTOR_ENDPOINT_NAME}")

# COMMAND ----------

try:
    indexes = vsc.list_indexes(VECTOR_ENDPOINT_NAME)
    index_list = (
        indexes.get("vector_indexes")
        or indexes.get("indexes")
        or []
    )
    index_names = {idx.get("name") for idx in index_list}

    if VECTOR_INDEX_NAME not in index_names:
        print(f"Creating index: {VECTOR_INDEX_NAME}")
        vsc.create_delta_sync_index(
            endpoint_name=VECTOR_ENDPOINT_NAME,
            index_name=VECTOR_INDEX_NAME,
            source_table_name=EMBEDDINGS_TABLE,
            pipeline_type="TRIGGERED",
            primary_key=PRIMARY_KEY,
            embedding_dimension=EMBEDDING_DIMENSION,
            embedding_vector_column=EMBEDDING_COLUMN,
        )
        print("Index creation triggered. It may take a few minutes to come ONLINE.")
    else:
        print(f"Index already exists: {VECTOR_INDEX_NAME}")
except Exception as exc:
    print("Detailed error:", str(exc))
    raise RuntimeError("Failed to create or validate Vector Search index.") from exc

print()
print("Vector Search setup complete.")
print(f"Endpoint: {VECTOR_ENDPOINT_NAME}")
print(f"Index   : {VECTOR_INDEX_NAME}")


# COMMAND ----------

import time

idx = vsc.get_index(VECTOR_ENDPOINT_NAME, VECTOR_INDEX_NAME)

deadline = time.time() + 30 * 60  # wait up to 30 min
last = None
while time.time() < deadline:
    info = idx.describe()
    state = (info.get("status") or {}).get("detailed_state") \
            or (info.get("status") or {}).get("ready") \
            or info.get("status")
    ready = (info.get("status") or {}).get("ready", False)
    indexed = (info.get("status") or {}).get("indexed_row_count")
    if state != last:
        print(f"index state: {state} | indexed_rows={indexed} | ready={ready}")
        last = state
    if ready:
        print("Index is READY for queries.")
        break
    time.sleep(20)
else:
    print("Index did not become READY in time; check Catalog Explorer for details.")