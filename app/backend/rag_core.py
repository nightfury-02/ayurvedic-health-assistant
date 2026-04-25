"""Reusable RAG core for AyurGenix.

This module contains the actual retrieval + generation pipeline, with no
framework dependencies. It can be called from:

* a Streamlit app (in-process, no HTTP) — the recommended deployment when
  Streamlit and the Databricks workspace share credentials (e.g. inside
  Databricks Apps).
* a FastAPI service (``databricks_notebooks/rag_pipeline/06_api_serving.py``)
  that simply wraps :func:`ask` in HTTP routes.
* a notebook or unit test.

Authentication is delegated to ``databricks.sdk.WorkspaceClient``, which
auto-discovers credentials from environment variables, ``DATABRICKS_HOST`` /
``DATABRICKS_TOKEN``, profile config, or the Databricks Apps runtime.

Public API:

* :data:`SUPPORTED_LANGUAGES` -- code -> human-readable name
* :func:`detect_language` / :func:`resolve_language`
* :func:`ask` -- the one-call entry point that does detect → translate →
  retrieve → generate → translate-back, returning an :class:`AskResult`.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---- Configuration ----------------------------------------------------------
# Override any of these from the environment without editing code.
CATALOG = os.getenv("AYURGENIX_CATALOG", "bricksiitm")
SCHEMA = os.getenv("AYURGENIX_SCHEMA", "ayurgenix")

VECTOR_ENDPOINT_NAME = os.getenv(
    "AYURGENIX_VECTOR_ENDPOINT", "ayurgenix-vs-endpoint"
)
VECTOR_INDEX_NAME = os.getenv(
    "AYURGENIX_VECTOR_INDEX",
    f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index",
)
EMBEDDING_ENDPOINT = os.getenv(
    "AYURGENIX_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
)
LLM_ENDPOINT = os.getenv(
    "AYURGENIX_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
)

DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 20
MMR_LAMBDA = 0.5
USE_HYBRID = True

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

logger = logging.getLogger("ayurgenix.rag_core")


# ---- Language detection / translation prompts -------------------------------
# (Inlined here so this module has no inter-package import gymnastics. Kept
# in sync with databricks_notebooks/rag_pipeline/language_utils.py.)

_SCRIPT_RANGES = [
    ("hi", "Hindi",     0x0900, 0x097F),
    ("bn", "Bengali",   0x0980, 0x09FF),
    ("pa", "Punjabi",   0x0A00, 0x0A7F),
    ("gu", "Gujarati",  0x0A80, 0x0AFF),
    ("or", "Odia",      0x0B00, 0x0B7F),
    ("ta", "Tamil",     0x0B80, 0x0BFF),
    ("te", "Telugu",    0x0C00, 0x0C7F),
    ("kn", "Kannada",   0x0C80, 0x0CFF),
    ("ml", "Malayalam", 0x0D00, 0x0D7F),
    ("ur", "Urdu",      0x0600, 0x06FF),
]

SUPPORTED_LANGUAGES: Dict[str, str] = {
    code: name for code, name, _, _ in _SCRIPT_RANGES
}
SUPPORTED_LANGUAGES["en"] = "English"


@dataclass(frozen=True)
class DetectedLanguage:
    code: str
    name: str
    is_english: bool


def detect_language(text: str) -> DetectedLanguage:
    if not text:
        return DetectedLanguage("en", "English", True)
    counts: Dict[Any, int] = {}
    for ch in text:
        cp = ord(ch)
        for code, name, start, end in _SCRIPT_RANGES:
            if start <= cp <= end:
                counts[(code, name)] = counts.get((code, name), 0) + 1
                break
    if not counts:
        return DetectedLanguage("en", "English", True)
    (code, name), _ = max(counts.items(), key=lambda kv: kv[1])
    return DetectedLanguage(code=code, name=name, is_english=False)


def resolve_language(code_or_name: Optional[str]) -> DetectedLanguage:
    if not code_or_name:
        return DetectedLanguage("en", "English", True)
    key = code_or_name.strip().lower()
    for code, name in SUPPORTED_LANGUAGES.items():
        if key == code.lower() or key == name.lower():
            return DetectedLanguage(code=code, name=name, is_english=(code == "en"))
    return DetectedLanguage("en", "English", True)


def _build_translate_messages(
    text: str, source_language_name: str, target_language_name: str
) -> List[Dict[str, str]]:
    system = (
        "You are a precise translator. Translate the user's message from "
        f"{source_language_name} to {target_language_name}. "
        "Preserve named entities, numbers, Sanskrit and Latin scientific "
        "terms (Vata, Pitta, Kapha, Ashwagandha, Withania somnifera, "
        "Triphala), and any bracketed citations such as [source_file, page] "
        "EXACTLY as written. Return ONLY the translation, with no preamble, "
        "no quotation marks, and no explanations."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


# ---- Result types -----------------------------------------------------------

@dataclass(frozen=True)
class Source:
    chunk_text: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class AskResult:
    answer: str
    language: str
    detected_language: str
    disclaimer: str = MEDICAL_DISCLAIMER
    sources: List[Source] = field(default_factory=list)


# ---- Lazy SDK clients (constructed on first use, then cached) ---------------

_workspace_client = None
_vector_search_client = None
_vector_index = None


def _normalize_databricks_host() -> None:
    """Ensure ``DATABRICKS_HOST`` carries the ``https://`` scheme.

    The Databricks Apps runtime sometimes injects the bare hostname
    (``dbc-xxxx.cloud.databricks.com``). The Python SDK tolerates that, but
    ``databricks-vectorsearch``'s ``VectorSearchClient`` concatenates the
    host with the API path and ``requests`` then rejects the result with
    ``MissingSchema: Invalid URL ... No scheme supplied``.
    """
    host = os.getenv("DATABRICKS_HOST", "").strip()
    if host and not host.startswith(("http://", "https://")):
        os.environ["DATABRICKS_HOST"] = f"https://{host}"


def _disambiguate_auth() -> None:
    """Resolve the SDK's "more than one authorization method" error.

    On Databricks Apps the runtime auto-injects ``DATABRICKS_CLIENT_ID`` and
    ``DATABRICKS_CLIENT_SECRET`` for the app's service principal. If a PAT
    is also configured (via ``DATABRICKS_TOKEN``), ``WorkspaceClient`` raises
    ``ValueError: more than one authorization method configured``. We pick a
    deterministic winner instead of failing: explicit ``DATABRICKS_AUTH_TYPE``
    wins, otherwise PAT beats OAuth (since the user-supplied PAT is the one
    they intended to use).
    """
    if os.getenv("DATABRICKS_AUTH_TYPE"):
        return
    has_pat = bool(os.getenv("DATABRICKS_TOKEN"))
    has_oauth = bool(os.getenv("DATABRICKS_CLIENT_ID")) and bool(
        os.getenv("DATABRICKS_CLIENT_SECRET")
    )
    if has_pat and has_oauth:
        os.environ["DATABRICKS_AUTH_TYPE"] = "pat"


def _get_workspace_client():
    global _workspace_client
    if _workspace_client is None:
        _normalize_databricks_host()
        _disambiguate_auth()
        from databricks.sdk import WorkspaceClient  # noqa: WPS433
        try:
            _workspace_client = WorkspaceClient()
        except Exception as exc:  # noqa: BLE001
            # The SDK raises InvalidInputException when none of its auth
            # discovery paths (PAT, M2M OAuth, profile, notebook) yielded
            # credentials. On Databricks Apps that means the app's service
            # principal credentials weren't injected — usually because the
            # required resources haven't been bound to the app yet.
            present = {
                k: bool(os.getenv(k))
                for k in (
                    "DATABRICKS_HOST",
                    "DATABRICKS_TOKEN",
                    "DATABRICKS_CLIENT_ID",
                    "DATABRICKS_CLIENT_SECRET",
                )
            }
            raise RuntimeError(
                "Databricks SDK could not authenticate. On Databricks Apps "
                "this usually means the app's service principal hasn't been "
                "granted access to the serving / vector-search endpoints yet "
                "(open the app in the workspace -> Resources -> add the LLM "
                "endpoint, embedding endpoint, vector-search endpoint, and "
                "vector-search index, then redeploy). Locally, set "
                "DATABRICKS_HOST + DATABRICKS_TOKEN. "
                f"Detected env vars: {present}. Underlying error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return _workspace_client


def _get_vector_index():
    global _vector_search_client, _vector_index
    if _vector_index is None:
        _normalize_databricks_host()
        _disambiguate_auth()
        from databricks.vector_search.client import VectorSearchClient  # noqa: WPS433
        _vector_search_client = VectorSearchClient()
        _vector_index = _vector_search_client.get_index(
            endpoint_name=VECTOR_ENDPOINT_NAME,
            index_name=VECTOR_INDEX_NAME,
        )
    return _vector_index


def _direct_rest_invoke(endpoint_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call a Databricks serving endpoint directly via HTTPS.

    This bypasses ``databricks-sdk`` entirely. Useful when the SDK has an
    internal serialization bug (e.g. the known
    ``AttributeError: 'dict' object has no attribute 'as_dict'`` from
    ``QueryEndpointResponse`` in some SDK versions).

    Returns the parsed JSON body as a plain ``dict``.
    """
    import requests  # noqa: WPS433

    _normalize_databricks_host()
    host = os.getenv("DATABRICKS_HOST", "").strip().rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    if not host:
        raise RuntimeError("DATABRICKS_HOST is not set; cannot invoke REST API.")
    if not token:
        raise RuntimeError(
            "DATABRICKS_TOKEN is not set; direct REST fallback requires a PAT."
        )
    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Direct REST call to '{endpoint_name}' returned HTTP "
            f"{resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


# ---- Embedding --------------------------------------------------------------

def embed_query(question: str) -> List[float]:
    """Embed via the serving endpoint directly (FM-API ``input=[...]`` schema,
    falling back to legacy ``inputs=[...]`` for custom MLflow endpoints)."""
    if not question or not question.strip():
        raise ValueError("Question is empty.")
    w = _get_workspace_client()
    last_exc: Optional[Exception] = None
    resp = None
    for kwargs in ({"input": [question]}, {"inputs": [question]}):
        try:
            resp = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if resp is None:
        # SDK call failed. Fall back to a direct REST call before giving up.
        try:
            data = _direct_rest_invoke(EMBEDDING_ENDPOINT, {"input": [question]})
        except Exception:  # noqa: BLE001
            try:
                data = _direct_rest_invoke(
                    EMBEDDING_ENDPOINT, {"inputs": [question]}
                )
            except Exception as rest_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Embedding endpoint '{EMBEDDING_ENDPOINT}' failed via "
                    f"both SDK and direct REST. SDK error: {last_exc!s}; "
                    f"REST error: {rest_exc!s}"
                ) from rest_exc
    else:
        data = _coerce_to_dict(resp)
    items = data.get("data") or data.get("predictions") or []
    if not items:
        raise RuntimeError("Embedding endpoint returned no data.")
    first = items[0]
    if isinstance(first, dict):
        embedding = (
            first.get("embedding") or first.get("vector") or first.get("output")
        )
    else:
        embedding = first
    if not embedding:
        raise RuntimeError("Embedding endpoint returned an empty vector.")
    return list(map(float, embedding))


# ---- Vector search + MMR re-ranking -----------------------------------------

def _decode_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        item = (
            dict(zip(column_names, row))
            if column_names and len(row) == len(column_names)
            else {f"col_{i}": v for i, v in enumerate(row)}
        )
        for score_key in ("score", "_score", "similarity_score"):
            if score_key in item:
                item["score"] = (
                    item.pop(score_key) if score_key != "score" else item[score_key]
                )
                break
        decoded.append(item)
    return decoded


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mmr_rerank(
    candidates: List[Dict[str, Any]],
    *,
    top_k: int,
    lambda_: float = MMR_LAMBDA,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates
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
        best_idx, best_score = -1, -math.inf
        for i in remaining:
            diversity_penalty = (
                max(_cosine(bow[i], bow[j]) for j in selected) if selected else 0.0
            )
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def _retrieve(
    query: str, top_k: int, source_filter: Optional[str]
) -> List[Dict[str, Any]]:
    index = _get_vector_index()
    fetch_k = max(DEFAULT_FETCH_K, top_k * 4)
    query_vec = embed_query(query)
    search_kwargs: Dict[str, Any] = dict(
        query_vector=query_vec,
        columns=["chunk_text", "source_file", "page_number"],
        num_results=fetch_k,
    )
    if USE_HYBRID:
        search_kwargs["query_text"] = query
        search_kwargs["query_type"] = "HYBRID"

    try:
        response = index.similarity_search(**search_kwargs)
    except Exception as exc:  # noqa: BLE001
        if USE_HYBRID:
            logger.warning("Hybrid search failed (%s); falling back to vector-only.", exc)
            search_kwargs.pop("query_text", None)
            search_kwargs.pop("query_type", None)
            response = index.similarity_search(**search_kwargs)
        else:
            raise

    candidates = _decode_results(response)

    # Source-type filter applied post-retrieval. Our chunk_id column uses the
    # convention "csv:..." or "pdf:..." set by the ingestion notebooks; we
    # request the chunk_id only when filtering, then over-fetch enough to
    # absorb the loss.
    if source_filter in {"csv", "pdf"}:
        candidates = [
            c for c in candidates
            if str(c.get("chunk_id") or c.get("source_file") or "").lower()
            .startswith(source_filter)
            or str(c.get("source_file") or "").lower().endswith(
                ".csv" if source_filter == "csv" else ".pdf"
            )
        ]

    return _mmr_rerank(candidates, top_k=top_k)


# ---- LLM call (chat completion) with multi-version fallbacks ---------------

def _coerce_to_dict(resp: Any) -> Dict[str, Any]:
    """Best-effort serialization to a plain dict.

    Tolerates the half-dozen shapes ``serving_endpoints.query`` /
    ``chat.completions.create`` can return across SDK versions:

    * already a ``dict``
    * Databricks SDK dataclass with ``.as_dict()``  (older SDKs)
    * Pydantic v1 model with ``.dict()``           (some 0.40+ versions)
    * Pydantic v2 model with ``.model_dump()``     (newest)
    * dataclass-ish object with ``__dict__``
    """
    if isinstance(resp, dict):
        return resp
    for attr in ("model_dump", "dict", "as_dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                value = fn()
                if isinstance(value, dict):
                    return value
            except Exception:  # noqa: BLE001
                continue
    return getattr(resp, "__dict__", {}) or {}


def _extract_answer(resp: Any) -> str:
    """Pull the assistant text out of any chat-completion response shape."""
    # 1. Typed-object path: resp.choices[0].message.content
    typed_choices = getattr(resp, "choices", None)
    if typed_choices:
        try:
            first = typed_choices[0]
        except (IndexError, KeyError, TypeError):
            first = None
        if first is not None:
            msg = getattr(first, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if content:
                return content
            if isinstance(first, dict):
                m = first.get("message") or {}
                if isinstance(m, dict) and m.get("content"):
                    return m["content"]
                if first.get("text"):
                    return first["text"]

    # 2. Dict-shaped paths.
    data = _coerce_to_dict(resp)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        msg = first.get("message") or {} if isinstance(first, dict) else {}
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
        if isinstance(first, dict) and first.get("text"):
            return first["text"]

    predictions = data.get("predictions")
    if isinstance(predictions, list) and predictions:
        pred = predictions[0]
        if isinstance(pred, dict):
            cand = (pred.get("candidates") or [None])[0]
            if isinstance(cand, dict) and cand.get("text"):
                return cand["text"]
            if pred.get("content"):
                return pred["content"]
        elif isinstance(pred, str):
            return pred

    # 3. Some Databricks Foundation Model responses use {"output": "..."}
    # or a top-level "content" string.
    for key in ("output", "content", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise ValueError(
        f"LLM returned no usable content. Raw response type: {type(resp).__name__}; "
        f"coerced dict keys: {list(data.keys())}"
    )


def _chat(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    w = _get_workspace_client()
    last_exc: Optional[Exception] = None

    # Path A: OpenAI-compatible client (newer SDK; cleanest typed response).
    open_ai_client = None
    try:
        open_ai_client = w.serving_endpoints.get_open_ai_client()
    except AttributeError:
        # Older databricks-sdk: get_open_ai_client doesn't exist. Fall through
        # to native query below.
        open_ai_client = None

    if open_ai_client is not None:
        try:
            resp = open_ai_client.chat.completions.create(
                model=LLM_ENDPOINT,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return _extract_answer(resp)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "OpenAI-compatible LLM call failed (%s: %s); falling back to native query.",
                type(exc).__name__, exc,
            )

    # Path B: native serving_endpoints.query, with two payload shapes for
    # cross-version compatibility.
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

    # Path C: direct REST call. Bypasses the SDK entirely; useful when the
    # SDK has an internal serialization bug ("'dict' object has no attribute
    # 'as_dict'") that no amount of response handling on our side can fix.
    try:
        data = _direct_rest_invoke(
            LLM_ENDPOINT,
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return _extract_answer(data)
    except Exception as exc:  # noqa: BLE001
        last_exc = exc

    raise RuntimeError(
        f"All LLM invocation paths failed for endpoint '{LLM_ENDPOINT}' "
        f"(OpenAI client, native query, direct REST). "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _translate_text(text: str, source_lang_name: str, target_lang_name: str) -> str:
    if not text or not text.strip():
        return ""
    if source_lang_name.lower() == target_lang_name.lower():
        return text
    return _chat(
        _build_translate_messages(text, source_lang_name, target_lang_name),
        max_tokens=768,
        temperature=0.0,
    ).strip()


def _detect_red_flag(question: str) -> Optional[str]:
    if _RED_FLAG_RE.search(question or ""):
        return (
            "Your question mentions symptoms that may require urgent medical "
            "attention. Please contact local emergency services or a qualified "
            "clinician immediately. This assistant cannot provide acute care "
            "guidance."
        )
    return None


def _build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c.get('source_file', 'unknown')}, "
        f"Page: {c.get('page_number')}] {c.get('chunk_text', '')}"
        for c in contexts
    ) or "(no context)"
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely (3-6 sentences), grounded strictly in the context, "
        "with inline citations of the form [source_file, page]."
    )


# ---- Public entry point -----------------------------------------------------

def ask(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    source_filter: Optional[str] = None,
    language: Optional[str] = None,
) -> AskResult:
    """Run the full RAG pipeline for one question.

    Parameters
    ----------
    question:
        The user's question, in any supported language.
    top_k:
        Number of context passages to keep after MMR re-ranking.
    source_filter:
        ``"csv"`` to restrict to herb-monograph CSV chunks, ``"pdf"`` to
        restrict to textbook PDF chunks, or ``None`` for both.
    language:
        Override for the response language. Code (``"te"``) or name
        (``"Telugu"``). If ``None``, the response uses the detected query
        language.
    """
    if not question or not question.strip():
        raise ValueError("question is required")

    detected = detect_language(question)
    target = resolve_language(language) if language else detected

    # Red-flag check on the raw input.
    red_flag_en = _detect_red_flag(question)
    if red_flag_en:
        try:
            answer = (
                red_flag_en if target.is_english
                else _translate_text(red_flag_en, "English", target.name)
            )
        except Exception:  # noqa: BLE001
            answer = red_flag_en
        return AskResult(
            answer=answer,
            language=target.code,
            detected_language=detected.code,
            sources=[],
        )

    english_question = (
        question if detected.is_english
        else _translate_text(question, detected.name, "English")
    )

    # Re-check red flags against the translated form.
    if not detected.is_english:
        rf2 = _detect_red_flag(english_question)
        if rf2:
            try:
                answer = (
                    rf2 if target.is_english
                    else _translate_text(rf2, "English", target.name)
                )
            except Exception:  # noqa: BLE001
                answer = rf2
            return AskResult(
                answer=answer,
                language=target.code,
                detected_language=detected.code,
                sources=[],
            )

    contexts = _retrieve(english_question, top_k, source_filter)

    if not contexts:
        msg_en = (
            "I could not find relevant context in the knowledge base for that "
            "question."
        )
        try:
            answer = (
                msg_en if target.is_english
                else _translate_text(msg_en, "English", target.name)
            )
        except Exception:  # noqa: BLE001
            answer = msg_en
        return AskResult(
            answer=answer,
            language=target.code,
            detected_language=detected.code,
            sources=[],
        )

    english_answer = _chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(english_question, contexts)},
        ],
        max_tokens=512,
        temperature=0.2,
    )

    final_answer = (
        english_answer if target.is_english
        else _translate_text(english_answer, "English", target.name)
    )

    sources = [
        Source(
            chunk_text=str(c.get("chunk_text") or ""),
            source_file=c.get("source_file"),
            page_number=int(c["page_number"]) if c.get("page_number") is not None else None,
            score=float(c["score"]) if c.get("score") is not None else None,
        )
        for c in contexts
    ]
    return AskResult(
        answer=final_answer,
        language=target.code,
        detected_language=detected.code,
        sources=sources,
    )
