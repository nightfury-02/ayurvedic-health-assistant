# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - API Serving (FastAPI)
# MAGIC
# MAGIC This notebook shows a production-friendly FastAPI app for RAG serving.
# MAGIC Endpoint:
# MAGIC - `POST /ask`
# MAGIC - Input: `{"question": "...", "top_k": 5}`
# MAGIC - Output: `{"answer": "...", "sources": [...]}`.

# COMMAND ----------

# MAGIC %pip install fastapi uvicorn databricks-sdk databricks-vectorsearch

# COMMAND ----------

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

app = FastAPI(title="AyurGenix RAG API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


def get_query_embedding(question: str):
    safe_query = question.replace("'", "\\'")
    row = spark.sql(
        f"SELECT ai_query('{EMBEDDING_ENDPOINT}', '{safe_query}') AS embedding"
    ).first()
    embedding = row["embedding"] if row else None
    if not embedding:
        raise ValueError("Failed to generate query embedding.")
    return embedding


def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    embedding = get_query_embedding(question)
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT_NAME,
        index_name=VECTOR_INDEX_NAME,
    )
    response = index.similarity_search(
        query_vector=embedding,
        columns=["chunk_text", "source_file", "page_number"],
        num_results=top_k,
    )
    rows = response.get("result", {}).get("data_array", [])
    out = []
    for r in rows:
        out.append(
            {
                "chunk_text": r[0] if len(r) > 0 else "",
                "source_file": r[1] if len(r) > 1 else None,
                "page_number": r[2] if len(r) > 2 else None,
                "score": r[-1] if len(r) > 0 else None,
            }
        )
    return out


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        [
            f"[Source: {c.get('source_file')}, Page: {c.get('page_number')}] {c.get('chunk_text')}"
            for c in contexts
        ]
    )
    return (
        "Answer the question using the context below:\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "If context is insufficient, mention that clearly."
    )


def call_llm(prompt: str) -> str:
    w = WorkspaceClient()
    resp = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    if resp.choices and len(resp.choices) > 0:
        return resp.choices[0].message.content
    raise ValueError("No answer returned by LLM endpoint.")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        contexts = retrieve(req.question, req.top_k)
        if not contexts:
            raise ValueError("No relevant chunks retrieved.")
        prompt = build_prompt(req.question, contexts)
        answer = call_llm(prompt)
        return AskResponse(answer=answer, sources=contexts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run locally in Databricks driver (for testing)

# COMMAND ----------

# MAGIC %sh
# MAGIC # uvicorn app:app --host 0.0.0.0 --port 8000
