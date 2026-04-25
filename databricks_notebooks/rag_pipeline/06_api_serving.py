# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - API Serving (FastAPI)
# MAGIC
# MAGIC Production-leaning FastAPI app for RAG serving:
# MAGIC - Direct embedding via the serving endpoint (no per-request Spark job)
# MAGIC - Hybrid retrieval (vector + keyword) with MMR re-ranking
# MAGIC - Manifest-based row decoding (no fragile positional access)
# MAGIC - System prompt + medical safety preamble
# MAGIC - Red-flag symptom guard
# MAGIC - Per-request timeout and structured error responses
# MAGIC
# MAGIC Endpoints:
# MAGIC - `GET /health`
# MAGIC - `POST /ask`  body: `{"question": "...", "top_k": 5, "source_filter": "csv|pdf|null"}`

# COMMAND ----------

# MAGIC %pip install fastapi uvicorn databricks-sdk databricks-vectorsearch openai numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib.util
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient


def _import_language_utils():
    """Locate and import the shared ``language_utils`` module."""
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

# ---- Configuration ----------------------------------------------------------

CATALOG = "bricksiitm"
SCHEMA = "ayurgenix"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 20
MMR_LAMBDA = 0.5
USE_HYBRID = True

REQUEST_TIMEOUT_S = 30  # hard ceiling per request stage

ALLOWED_ORIGINS = ["*"]  # tighten to your Streamlit/app origin in production

MEDICAL_DISCLAIMER = (
    "This response summarizes traditional Ayurvedic guidance from the indexed "
    "sources. It is not medical advice. Consult a qualified practitioner for "
    "personal health decisions."
)

SYSTEM_PROMPT = (
    "You are an Ayurveda knowledge assistant. Answer the user's question "
    "using ONLY the provided context. Cite sources inline as "
    "[source_file, page]. If the context is insufficient or off-topic, say "
    "so explicitly instead of guessing. Do not provide diagnoses, dosages, "
    "or treatment recommendations as medical advice; frame suggestions as "
    "traditional Ayurvedic guidance and recommend consulting a qualified "
    "practitioner for personal health decisions."
)

# Symptom phrases that should short-circuit to an emergency-care message.
RED_FLAG_PATTERNS = [
    r"\bchest pain\b",
    r"\bdifficulty breathing\b|\bshortness of breath\b",
    r"\bsuicid\w*\b|\bself.?harm\b",
    r"\bsevere bleeding\b|\bheavy bleeding\b",
    r"\bunconscious\b|\bfainting\b",
    r"\bstroke\b|\bnumbness on one side\b|\bfacial droop\b",
    r"\bpregnan\w+\b.*\b(bleed|pain|cramp)\w*\b",
]
_RED_FLAG_RE = re.compile("|".join(RED_FLAG_PATTERNS), re.IGNORECASE)

logger = logging.getLogger("ayurgenix.rag")
logger.setLevel(logging.INFO)

# ---- App --------------------------------------------------------------------

app = FastAPI(title="AyurGenix RAG API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=8)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    source_filter: Optional[Literal["csv", "pdf"]] = None
    # Optional ISO-like code (e.g. "te", "hi") or human name ("Telugu") for the
    # response. If omitted the service auto-detects from the query script.
    language: Optional[str] = Field(
        default=None,
        description=(
            "Language code or name for the response. Leave null to auto-detect "
            "from the query. Supported codes: "
            + ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        ),
    )


class Source(BaseModel):
    chunk_text: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    language: str = Field(..., description="Language code of the returned answer.")
    detected_language: str = Field(
        ..., description="Language code detected from the user's query."
    )
    disclaimer: str
    sources: List[Source]


# ---- Helpers ----------------------------------------------------------------

def _run_with_timeout(fn, *args, timeout: int = REQUEST_TIMEOUT_S, **kwargs):
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Operation exceeded {timeout}s timeout") from exc


def _detect_red_flag(question: str) -> Optional[str]:
    if _RED_FLAG_RE.search(question or ""):
        return (
            "Your question mentions symptoms that may require urgent medical "
            "attention. Please contact local emergency services or a qualified "
            "clinician immediately. This assistant cannot provide acute care "
            "guidance."
        )
    return None


def embed_query(question: str) -> List[float]:
    """Embed via the serving endpoint directly.

    Foundation Model API embedding endpoints (e.g. ``databricks-gte-large-en``)
    follow the OpenAI embeddings schema and expect ``input=[...]``. Legacy
    MLflow-style endpoints expect ``inputs=[...]``. Try the FM-API form first
    and fall back to the legacy form for compatibility.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")
    w = WorkspaceClient()
    last_exc: Optional[Exception] = None
    resp = None
    for kwargs in ({"input": [question]}, {"inputs": [question]}):
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
    lambda_: float = MMR_LAMBDA,
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
        best_idx, best_score = -1, -math.inf
        for i in remaining:
            penalty = 0.0 if not selected else max(_cosine(bow[i], bow[j]) for j in selected)
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * penalty
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def retrieve(question: str, top_k: int, source_filter: Optional[str]) -> List[Dict[str, Any]]:
    embedding = embed_query(question)
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT_NAME,
        index_name=VECTOR_INDEX_NAME,
    )
    search_kwargs: Dict[str, Any] = dict(
        query_vector=embedding,
        columns=["chunk_text", "source_file", "page_number", "source_type"],
        num_results=DEFAULT_FETCH_K,
    )
    if source_filter:
        search_kwargs["filters"] = {"source_type": source_filter}
    if USE_HYBRID:
        search_kwargs["query_text"] = question
        search_kwargs["query_type"] = "HYBRID"

    try:
        response = index.similarity_search(**search_kwargs)
    except Exception as exc:
        if USE_HYBRID:
            logger.warning("Hybrid search failed (%s); falling back to vector-only.", exc)
            search_kwargs.pop("query_text", None)
            search_kwargs.pop("query_type", None)
            response = index.similarity_search(**search_kwargs)
        else:
            raise

    candidates = decode_results(response)
    return mmr_rerank(candidates, top_k=top_k)


def build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c.get('source_file', 'unknown')}, Page: {c.get('page_number')}] {c.get('chunk_text', '')}"
        for c in contexts
    ) or "(no context)"
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely (3-6 sentences), grounded strictly in the context, "
        "with inline citations of the form [source_file, page]."
    )


def _extract_answer(resp: Any) -> str:
    """Pull the assistant message out of a chat-completions response.

    Tolerates both the OpenAI typed-object shape and dict shapes from older
    ``serving_endpoints.query`` versions.
    """
    choices = getattr(resp, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if content:
            return content

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
    """Generic chat-completion call with multi-version fallbacks."""
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
        pass  # Older SDK; fall through.
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OpenAI-compatible client failed (%s: %s); falling back to native query.",
            type(exc).__name__,
            exc,
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
    if not text or not text.strip():
        return ""
    if source_language_name.lower() == target_language_name.lower():
        return text
    messages = build_translate_messages(text, source_language_name, target_language_name)
    return _chat(messages, max_tokens=768, temperature=0.0).strip()


def call_llm(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Generate an English answer grounded in the retrieved context."""
    user_prompt = build_user_prompt(question, contexts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return _chat(messages, max_tokens=512, temperature=0.2)


# ---- Routes -----------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/languages")
def languages() -> Dict[str, str]:
    """Return supported language code -> human name."""
    return SUPPORTED_LANGUAGES


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # 1. Resolve languages.
    detected = detect_language(req.question)
    target = resolve_language(req.language) if req.language else detected

    # 2. Red-flag check happens against the *original* question (regex is
    # English-only, so for non-English questions we also check the English
    # translation later, after we have it). We translate the canned response
    # into the target language so the user understands it.
    red_flag = _detect_red_flag(req.question)
    if red_flag:
        try:
            localized = translate_text(red_flag, "English", target.name) if not target.is_english else red_flag
        except Exception:  # noqa: BLE001
            localized = red_flag
        return AskResponse(
            answer=localized,
            language=target.code,
            detected_language=detected.code,
            disclaimer=MEDICAL_DISCLAIMER,
            sources=[],
        )

    # 3. Translate the question to English for retrieval/generation.
    try:
        if detected.is_english:
            english_question = req.question
        else:
            english_question = _run_with_timeout(
                translate_text, req.question, detected.name, "English"
            )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Query translation timed out.")
    except Exception:
        logger.exception("Query translation failed")
        raise HTTPException(status_code=502, detail="Query translation failed.")

    # Re-run red-flag check against the English form for non-English queries.
    if not detected.is_english:
        rf2 = _detect_red_flag(english_question)
        if rf2:
            try:
                localized = translate_text(rf2, "English", target.name) if not target.is_english else rf2
            except Exception:  # noqa: BLE001
                localized = rf2
            return AskResponse(
                answer=localized,
                language=target.code,
                detected_language=detected.code,
                disclaimer=MEDICAL_DISCLAIMER,
                sources=[],
            )

    # 4. Retrieve.
    try:
        contexts = _run_with_timeout(
            retrieve, english_question, req.top_k, req.source_filter
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Retrieval timed out.")
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail="Retrieval failed.")

    if not contexts:
        msg_en = "I could not find relevant context in the knowledge base for that question."
        try:
            answer = translate_text(msg_en, "English", target.name) if not target.is_english else msg_en
        except Exception:  # noqa: BLE001
            answer = msg_en
        return AskResponse(
            answer=answer,
            language=target.code,
            detected_language=detected.code,
            disclaimer=MEDICAL_DISCLAIMER,
            sources=[],
        )

    # 5. Generate the English answer.
    try:
        english_answer = _run_with_timeout(call_llm, english_question, contexts)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="LLM generation timed out.")
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail="LLM generation failed.")

    # 6. Translate the answer back if needed.
    if target.is_english:
        final_answer = english_answer
    else:
        try:
            final_answer = _run_with_timeout(
                translate_text, english_answer, "English", target.name
            )
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Answer translation timed out.")
        except Exception:
            logger.exception("Answer translation failed")
            # Degrade gracefully: return the English answer rather than 5xx.
            final_answer = english_answer

    sources = [
        Source(
            chunk_text=str(c.get("chunk_text") or ""),
            source_file=c.get("source_file"),
            page_number=int(c["page_number"]) if c.get("page_number") is not None else None,
            score=float(c["score"]) if c.get("score") is not None else None,
        )
        for c in contexts
    ]
    return AskResponse(
        answer=final_answer,
        language=target.code,
        detected_language=detected.code,
        disclaimer=MEDICAL_DISCLAIMER,
        sources=sources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run locally in Databricks driver (for testing)

# COMMAND ----------

# MAGIC %sh
# MAGIC # uvicorn app:app --host 0.0.0.0 --port 8000