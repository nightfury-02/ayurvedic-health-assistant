# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - RAG Pipeline
# MAGIC
# MAGIC End-to-end flow:
# MAGIC 1. User query
# MAGIC 2. Direct embedding via the serving endpoint (no Spark roundtrip)
# MAGIC 3. Hybrid search against the vector index (vector + keyword)
# MAGIC 4. MMR re-ranking for diversity
# MAGIC 5. Construct grounded prompt with system + user roles
# MAGIC 6. Call LLM with a medical safety preamble
# MAGIC 7. Return final answer + sources

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch databricks-sdk openai numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


# DBTITLE 1,Imports and configuration
import importlib.util
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient


def _import_language_utils():
    """Import the shared language_utils module from the notebook workspace.

    Mirrors the discovery pattern used in the ingestion notebooks for
    ``chunking_utils.py`` so this works whether the notebook runs as part of a
    Databricks Repo (importable as a package) or as a standalone workspace
    notebook (located by walking up to find the file on disk).
    """
    try:
        from databricks_notebooks.rag_pipeline.language_utils import (  # noqa: WPS433
            DetectedLanguage,
            SUPPORTED_LANGUAGES,
            build_translate_messages,
            detect_language,
            resolve_language,
        )
        return {
            "DetectedLanguage": DetectedLanguage,
            "SUPPORTED_LANGUAGES": SUPPORTED_LANGUAGES,
            "build_translate_messages": build_translate_messages,
            "detect_language": detect_language,
            "resolve_language": resolve_language,
        }
    except Exception:  # noqa: BLE001
        current = os.getcwd()
        while True:
            candidate = os.path.join(
                current, "databricks_notebooks", "rag_pipeline", "language_utils.py"
            )
            if os.path.exists(candidate):
                spec = importlib.util.spec_from_file_location("language_utils", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                return {
                    "DetectedLanguage": module.DetectedLanguage,
                    "SUPPORTED_LANGUAGES": module.SUPPORTED_LANGUAGES,
                    "build_translate_messages": module.build_translate_messages,
                    "detect_language": module.detect_language,
                    "resolve_language": module.resolve_language,
                }
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        raise ImportError("Unable to locate shared language_utils.py")


_lang = _import_language_utils()
DetectedLanguage = _lang["DetectedLanguage"]
SUPPORTED_LANGUAGES = _lang["SUPPORTED_LANGUAGES"]
build_translate_messages = _lang["build_translate_messages"]
detect_language = _lang["detect_language"]
resolve_language = _lang["resolve_language"]


CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

TOP_K = 5
FETCH_K = 20
MMR_LAMBDA = 0.5
USE_HYBRID = True

# Set USER_QUERY in any supported regional language; the pipeline will detect
# the script automatically and translate via the LLM. Examples:
#   "నాకు బాగా నిద్ర పట్టడం లేదు, ఆయుర్వేదంలో ఏం చేయాలి?"  (Telugu)
#   "अच्छी नींद के लिए आयुर्वेद क्या सुझाता है?"            (Hindi)
#   "நல்ல தூக்கத்திற்கு ஆயுர்வேதம் என்ன பரிந்துரைக்கிறது?"  (Tamil)
USER_QUERY = "నాకు బాగా నిద్ర పట్టడం లేదు, ఆయుర్వేదంలో ఏం చేయాలి?" 

# Force a specific language for the answer instead of auto-detecting from the
# query script. Use a code from SUPPORTED_LANGUAGES (e.g. "te"), a name
# (e.g. "Telugu"), or leave as None to auto-detect.
RESPONSE_LANGUAGE: Optional[str] = None

SYSTEM_PROMPT = (
    "You are an Ayurveda knowledge assistant. Answer the user's question "
    "using ONLY the provided context. Cite sources inline as "
    "[source_file, page]. If the context is insufficient or off-topic, say "
    "so explicitly instead of guessing. Do not provide diagnoses, dosages, "
    "or treatment recommendations as medical advice; frame suggestions as "
    "traditional Ayurvedic guidance and recommend consulting a qualified "
    "practitioner for personal health decisions."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------




def embed_query(query: str) -> List[float]:
    """Embed via the serving endpoint directly.

    Foundation Model API embedding endpoints (e.g. ``databricks-gte-large-en``)
    follow the OpenAI embeddings schema and expect ``input=[...]``. Legacy
    MLflow-style endpoints expect ``inputs=[...]``. Try the FM-API form first
    and fall back to the legacy form for compatibility.
    """
    if not query or not query.strip():
        raise ValueError("Query is empty.")
    w = WorkspaceClient()
    last_exc: Optional[Exception] = None
    resp = None
    for kwargs in ({"input": [query]}, {"inputs": [query]}):
        try:
            resp = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if resp is None:
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
    """Map raw vector-search rows to dicts using the response manifest."""
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
            item = {f"col_{i}": v for i, v in enumerate(row)}
        for score_key in ("score", "_score", "similarity_score"):
            if score_key in item and score_key != "score":
                item["score"] = item.pop(score_key)
                break
        decoded.append(item)
    return decoded


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)


def mmr_rerank(
    candidates: List[Dict[str, Any]],
    top_k: int,
    lambda_: float = 0.5,
) -> List[Dict[str, Any]]:
    if not candidates or len(candidates) <= top_k:
        return candidates

    docs = [str(c.get("chunk_text", "")) for c in candidates]
    vocab: Dict[str, int] = {}
    for doc in docs:
        for tok in doc.lower().split():
            vocab.setdefault(tok, len(vocab))
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
                penalty = 0.0
            else:
                penalty = max(_cosine(bow[i], bow[j]) for j in selected)
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * penalty
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def retrieve_context(query: str, k: int = TOP_K, fetch_k: int = FETCH_K) -> List[Dict[str, Any]]:
    embedding = embed_query(query)
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT_NAME,
        index_name=VECTOR_INDEX_NAME,
    )

    search_kwargs: Dict[str, Any] = dict(
        query_vector=embedding,
        columns=["chunk_text", "source_file", "page_number"],
        num_results=fetch_k,
    )
    if USE_HYBRID:
        search_kwargs["query_text"] = query
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
    return mmr_rerank(candidates, top_k=k, lambda_=MMR_LAMBDA)


def build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_lines = []
    for c in contexts:
        context_lines.append(
            f"[Source: {c.get('source_file', 'unknown')}, "
            f"Page: {c.get('page_number')}] {c.get('chunk_text', '')}"
        )
    context_block = "\n\n".join(context_lines) if context_lines else "(no context)"
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely (3-6 sentences), grounded strictly in the context, "
        "with inline citations of the form [source_file, page]."
    )


def _extract_answer(resp: Any) -> str:
    """Pull the assistant message out of a chat-completions response.

    Handles both the OpenAI-style typed object (``resp.choices[0].message.content``)
    and the dict shape returned by older ``serving_endpoints.query`` versions
    (``{"choices": [{"message": {"content": ...}}], ...}`` or
    ``{"predictions": [{"candidates": [{"text": ...}]}]}``).
    """
    # Typed object path (OpenAI client, newer SDKs).
    choices = getattr(resp, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if content:
            return content

    # Dict path.
    data = resp.as_dict() if hasattr(resp, "as_dict") else (
        resp if isinstance(resp, dict) else {}
    )
    if isinstance(data.get("choices"), list) and data["choices"]:
        msg = data["choices"][0].get("message") or {}
        if msg.get("content"):
            return msg["content"]
        if data["choices"][0].get("text"):
            return data["choices"][0]["text"]
    if isinstance(data.get("predictions"), list) and data["predictions"]:
        pred = data["predictions"][0]
        if isinstance(pred, dict):
            cand = (pred.get("candidates") or [None])[0]
            if isinstance(cand, dict) and cand.get("text"):
                return cand["text"]
            if pred.get("content"):
                return pred["content"]
        elif isinstance(pred, str):
            return pred

    raise ValueError(f"LLM returned no usable content. Raw response: {data!r}")


def _chat(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Run a chat-completion call against the Databricks LLM endpoint.

    Tries the OpenAI-compatible client first, then falls back to the native
    ``serving_endpoints.query()`` API in two forms (with and without
    ``extra_params``) so the same code works across SDK versions.
    """
    w = WorkspaceClient()

    try:
        client = w.serving_endpoints.get_open_ai_client()
        resp = client.chat.completions.create(
            model=LLM_ENDPOINT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _extract_answer(resp)
    except AttributeError:
        pass  # Older SDK without get_open_ai_client; fall through.
    except Exception as exc:  # noqa: BLE001
        print(
            f"OpenAI-compatible client failed ({type(exc).__name__}: {exc}); "
            "falling back to serving_endpoints.query()."
        )

    last_exc: Optional[Exception] = None
    for kwargs in (
        {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        {
            "messages": messages,
            "extra_params": {"max_tokens": max_tokens, "temperature": temperature},
        },
    ):
        try:
            resp = w.serving_endpoints.query(name=LLM_ENDPOINT, **kwargs)
            return _extract_answer(resp)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(
        f"All LLM invocation paths failed for endpoint '{LLM_ENDPOINT}'. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def translate_text(text: str, source_language_name: str, target_language_name: str) -> str:
    """Translate ``text`` between two languages using the LLM endpoint.

    Returns the input unchanged if source and target match, or if the input
    is blank. A short ``max_tokens`` budget keeps translation cheap; we use
    ``temperature=0.0`` for deterministic, faithful output.
    """
    if not text or not text.strip():
        return ""
    if source_language_name.lower() == target_language_name.lower():
        return text
    messages = build_translate_messages(text, source_language_name, target_language_name)
    return _chat(messages, max_tokens=768, temperature=0.0).strip()


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Generate an English answer grounded in the retrieved context."""
    user_prompt = build_user_prompt(question, contexts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return _chat(messages, max_tokens=512, temperature=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute RAG pipeline

# COMMAND ----------


# COMMAND ----------

# 1. Determine the user's language (auto-detect from script unless overridden).
input_lang = detect_language(USER_QUERY)
output_lang = (
    resolve_language(RESPONSE_LANGUAGE) if RESPONSE_LANGUAGE else input_lang
)

print(f"Detected query language : {input_lang.name} ({input_lang.code})")
print(f"Response language       : {output_lang.name} ({output_lang.code})")

# 2. Translate the query to English for retrieval (KB is English-only).
if input_lang.is_english:
    english_query = USER_QUERY
else:
    try:
        english_query = translate_text(USER_QUERY, input_lang.name, "English")
    except Exception as exc:
        raise RuntimeError(
            f"Query translation ({input_lang.name} -> English) failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(f"\nEnglish query (for retrieval):\n  {english_query}")

# 3. Retrieve grounded context.
try:
    contexts = retrieve_context(english_query, TOP_K, FETCH_K)
except Exception as exc:
    raise RuntimeError(
        f"Retrieval step failed: {type(exc).__name__}: {exc}"
    ) from exc

if not contexts:
    raise ValueError("No context retrieved from vector index.")

# 4. Generate an English answer from the context.
try:
    english_answer = generate_answer(english_query, contexts)
except Exception as exc:
    raise RuntimeError(
        f"LLM generation step failed: {type(exc).__name__}: {exc}"
    ) from exc

# 5. Translate the answer back to the user's language (if needed).
if output_lang.is_english:
    final_answer = english_answer
else:
    try:
        final_answer = translate_text(english_answer, "English", output_lang.name)
    except Exception as exc:
        raise RuntimeError(
            f"Answer translation (English -> {output_lang.name}) failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

print("\nQuestion:")
print(USER_QUERY)
print(f"\nAnswer ({output_lang.name}):")
print(final_answer)
print("\nSources:")
for c in contexts:
    print(f"- {c.get('source_file')} p.{c.get('page_number')} (score={c.get('score')})")
