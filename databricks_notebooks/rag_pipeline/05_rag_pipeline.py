# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - RAG Pipeline
# MAGIC
# MAGIC End-to-end flow:
# MAGIC 1. User query
# MAGIC 2. Retrieve top-k chunks from vector index
# MAGIC 3. Construct grounded prompt
# MAGIC 4. Call LLM
# MAGIC 5. Return final answer

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

TOP_K = 5
USER_QUERY = "How does Ayurveda recommend improving sleep quality?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def get_query_embedding(query: str):
    safe_query = query.replace("'", "\\'")
    result = spark.sql(
        f"SELECT ai_query('{EMBEDDING_ENDPOINT}', '{safe_query}') AS embedding"
    ).first()
    embedding = result["embedding"] if result else None
    if not embedding:
        raise ValueError("Query embedding is empty.")
    return embedding


def retrieve_context(query: str, k: int = TOP_K):
    embedding = get_query_embedding(query)
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT_NAME,
        index_name=VECTOR_INDEX_NAME,
    )
    response = index.similarity_search(
        query_vector=embedding,
        columns=["chunk_text", "source_file", "page_number"],
        num_results=k,
    )
    return response.get("result", {}).get("data_array", [])


def build_prompt(question: str, rows):
    context_lines = []
    for row in rows:
        chunk_text = row[0] if len(row) > 0 else ""
        source_file = row[1] if len(row) > 1 else "unknown"
        page_number = row[2] if len(row) > 2 else None
        context_lines.append(
            f"[Source: {source_file}, Page: {page_number}] {chunk_text}"
        )
    context_block = "\n\n".join(context_lines)
    return (
        "Answer the question using the context below:\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "If context is insufficient, say so clearly."
    )


def generate_answer(prompt: str):
    w = WorkspaceClient()
    resp = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    if resp.choices and len(resp.choices) > 0:
        return resp.choices[0].message.content
    raise ValueError("LLM returned no choices.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute RAG pipeline

# COMMAND ----------

try:
    retrieval_rows = retrieve_context(USER_QUERY, TOP_K)
except Exception as exc:
    raise RuntimeError("Retrieval step failed.") from exc

if not retrieval_rows:
    raise ValueError("No context retrieved from vector index.")

prompt = build_prompt(USER_QUERY, retrieval_rows)

try:
    answer = generate_answer(prompt)
except Exception as exc:
    raise RuntimeError("LLM generation step failed.") from exc

print("Question:")
print(USER_QUERY)
print("\nAnswer:")
print(answer)
