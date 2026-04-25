# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Retrieval Pipeline
# MAGIC
# MAGIC Flow:
# MAGIC 1. Accept user query
# MAGIC 2. Convert query to embedding (direct call to the serving endpoint)
# MAGIC 3. Hybrid (vector + keyword) similarity search against the index
# MAGIC 4. MMR re-rank for diversity
# MAGIC 5. Return `chunk_text`, source, page, and similarity score

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch databricks-sdk numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

TOP_K = 5
FETCH_K = 20  # over-fetch then re-rank with MMR
MMR_LAMBDA = 0.5
USE_HYBRID = True  # set False if your index does not support hybrid query

USER_QUERY = "What are the Ayurvedic recommendations for digestion and immunity?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers: query embedding, manifest-aware decoding, MMR

# COMMAND ----------



import math
from typing import Any, Dict, List, Optional

import numpy as np

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient


def embed_query(question: str) -> List[float]:
    """Embed via the serving endpoint directly. No Spark roundtrip per query.

    Foundation Model API embedding endpoints (e.g. ``databricks-gte-large-en``,
    ``databricks-bge-large-en``) follow the OpenAI embeddings schema and expect
    ``input=[...]``. Legacy MLflow-style endpoints expect ``inputs=[...]``.
    We try the FM-API form first and fall back to the legacy form.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")
    w = WorkspaceClient()
    last_exc: Optional[Exception] = None
    for kwargs in ({"input": [question]}, {"inputs": [question]}):
        try:
            resp = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    else:
        raise RuntimeError(
            f"Embedding endpoint '{EMBEDDING_ENDPOINT}' rejected both "
            f"'input' and 'inputs' payloads. Last error: {last_exc!s}"
        )

    data = resp.as_dict() if hasattr(resp, "as_dict") else dict(resp)
    items = data.get("data") or data.get("predictions") or []
    if not items:
        raise RuntimeError("Embedding endpoint returned no data.")
    first = items[0]
    if isinstance(first, dict):
        embedding = first.get("embedding") or first.get("vector") or first.get("output")
    else:
        embedding = first
    if not embedding:
        raise RuntimeError("Embedding endpoint returned an empty vector.")
    return list(map(float, embedding))


def decode_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map raw vector-search rows to dicts using the response manifest.

    Avoids the fragile positional `row[0]` / `row[-1]` decoding pattern, which
    can return chunk text in place of the score when only one column is requested.
    """
    result = response.get("result", {}) or {}
    manifest_columns = (
        result.get("manifest", {}).get("columns")
        or response.get("manifest", {}).get("columns")
        or []
    )
    column_names = [c.get("name") for c in manifest_columns if c.get("name")]
    rows = result.get("data_array") or []
    decoded: List[Dict[str, Any]] = []
    for row in rows:
        if column_names and len(row) == len(column_names):
            item = dict(zip(column_names, row))
        else:
            # Fallback when manifest is missing: the score is the appended last
            # column relative to the requested columns list.
            item = {f"col_{i}": v for i, v in enumerate(row)}
        # Normalize the score key across SDK versions.
        for score_key in ("score", "_score", "similarity_score"):
            if score_key in item:
                item["score"] = item.pop(score_key) if score_key != "score" else item[score_key]
                break
        decoded.append(item)
    return decoded


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_rerank(
    query_vec: List[float],
    candidates: List[Dict[str, Any]],
    top_k: int,
    lambda_: float = 0.5,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance re-ranking.

    Uses the candidate similarity scores as the relevance signal and the
    cosine similarity between candidate chunk_text bag-of-words as a cheap
    diversity proxy. This avoids re-embedding chunks at query time.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    # Cheap diversity proxy: bag-of-words vectors over the union vocabulary.
    docs = [str(c.get("chunk_text", "")) for c in candidates]
    vocab: Dict[str, int] = {}
    for doc in docs:
        for tok in doc.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    if not vocab:
        return candidates[:top_k]

    bow = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        for tok in doc.lower().split():
            bow[i, vocab[tok]] += 1.0

    relevance = np.array(
        [float(c.get("score") or 0.0) for c in candidates], dtype=np.float32
    )
    if relevance.max() > 0:
        relevance = relevance / relevance.max()

    selected: List[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_idx = -1
        best_score = -math.inf
        for i in remaining:
            if not selected:
                diversity_penalty = 0.0
            else:
                sims = [_cosine(bow[i], bow[j]) for j in selected]
                diversity_penalty = max(sims) if sims else 0.0
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve

# COMMAND ----------


# COMMAND ----------

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=VECTOR_ENDPOINT_NAME,
    index_name=VECTOR_INDEX_NAME,
)

query_embedding = embed_query(USER_QUERY)

search_kwargs: Dict[str, Any] = dict(
    query_vector=query_embedding,
    columns=["chunk_text", "source_file", "page_number"],
    num_results=FETCH_K,
)
if USE_HYBRID:
    search_kwargs["query_text"] = USER_QUERY
    search_kwargs["query_type"] = "HYBRID"

try:
    response = index.similarity_search(**search_kwargs)
except Exception as exc:
    if USE_HYBRID:
        print(f"Hybrid search failed ({exc}); falling back to vector-only.")
        search_kwargs.pop("query_text", None)
        search_kwargs.pop("query_type", None)
        response = index.similarity_search(**search_kwargs)
    else:
        raise

candidates = decode_results(response)
results = mmr_rerank(query_embedding, candidates, top_k=TOP_K, lambda_=MMR_LAMBDA)

if not results:
    print("No retrieval results found.")
else:
    print(f"Top {len(results)} retrieval results (after MMR):")
    for r in results:
        print({
            "chunk_text": (r.get("chunk_text") or "")[:200],
            "source_file": r.get("source_file"),
            "page_number": r.get("page_number"),
            "score": r.get("score"),
        })

# COMMAND ----------

# DBTITLE 1,Monitor index sync progress
import time

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=VECTOR_ENDPOINT_NAME,
    index_name=VECTOR_INDEX_NAME,
)

status = index.describe()["status"]

print("\n📊 Current Status:")
print(f"  State: {status['detailed_state']}")
print(f"  Ready: {status['ready']}")
print(f"  Indexed Rows: {status.get('indexed_row_count', 0)}")
print(f"  Message: {status.get('message', 'N/A')}")

if status["ready"]:
    print("\n✅ Index is ONLINE and ready for queries!")
else:
    print("\n⏳ Index is still provisioning. Re-run this cell to check progress.")

# COMMAND ----------

