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

# DBTITLE 1,Install required packages
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from typing import Dict, List, Optional

import requests
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

CATALOG = "ayurveda_assistant"
SCHEMA = "ingestion"

VECTOR_ENDPOINT_NAME = "ayurgenix-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.knowledge_base_embeddings_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

TOP_K = 5
USER_QUERY = "According to Ayurvedic what are uses of ginger?"
MAX_CONTEXT_CHARS = 5000

SARVAM_API_URL = os.getenv("SARVAM_API_URL", "https://api.sarvam.ai/translate")
SARVAM_API_KEY = dbutils.secrets.get(
    scope="ayurgenix-scope",
    key="sarvam_api_key"
)


SUPPORTED_LANGS = {"en", "hi", "ta", "te"}
LANG_NAME_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def get_query_embedding(query: str):
    safe_query = query.replace("'", "''")
    result = spark.sql(
        f"SELECT ai_query('{EMBEDDING_ENDPOINT}', '{safe_query}') AS embedding"
    ).first()
    embedding = result["embedding"] if result else None
    if not embedding:
        raise ValueError("Query embedding is empty.")
    return embedding


def normalize_lang_code(code: str) -> str:
    if not code:
        return "en"
    normalized = str(code).strip().lower()
    return LANG_NAME_TO_CODE.get(normalized, normalized)


def detect_language(text: str) -> str:
    content = str(text or "")
    if not content.strip():
        return "en"

    for ch in content:
        codepoint = ord(ch)
        if 0x0900 <= codepoint <= 0x097F:
            return "hi"
        if 0x0B80 <= codepoint <= 0x0BFF:
            return "ta"
        if 0x0C00 <= codepoint <= 0x0C7F:
            return "te"
    return "en"

def to_sarvam_lang(code: str) -> str:
    code = (code or "en").lower()
    mapping = {
        "en": "en-IN",
        "hi": "hi-IN",
        "ta": "ta-IN",
        "te": "te-IN",
    }
    return mapping.get(code, "en-IN")


def _sarvam_headers() -> Dict[str, str]:
    # Keep both header styles for compatibility across Sarvam API versions.
    return {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
        "Authorization": f"Bearer {SARVAM_API_KEY}",
    }


def _extract_translated_text(response_json: Dict, fallback_text: str) -> str:
    if not isinstance(response_json, dict):
        return fallback_text
    if response_json.get("translated_text"):
        return response_json["translated_text"]
    if response_json.get("output"):
        return response_json["output"]
    if response_json.get("translation"):
        return response_json["translation"]
    data = response_json.get("data")
    if isinstance(data, dict):
        return (
            data.get("translated_text")
            or data.get("output")
            or data.get("translation")
            or fallback_text
        )
    return fallback_text


def _lang_candidates(code: str) -> List[str]:
    code = (code or "en").lower()
    locale = to_sarvam_lang(code)
    return [locale, code]

def _translate_with_sarvam(text: str, source_lang: str, target_lang: str) -> str:
    if not text:
        return text
    if source_lang == target_lang:
        return text
    if not SARVAM_API_KEY:
        print("[DEBUG][RAG] SARVAM_API_KEY not set. Returning original text.")
        return text

    headers = _sarvam_headers()
    source_candidates = _lang_candidates(source_lang)
    target_candidates = _lang_candidates(target_lang)

    # Try multiple payload schemas because Sarvam API versions vary.
    payload_templates = [
        lambda src, tgt: {
            "input": text,
            "source_language_code": src,
            "target_language_code": tgt,
            "mode": "formal",
            "model": "mayura:v1",
        },
        lambda src, tgt: {
            "input": text,
            "source_language_code": src,
            "target_language_code": tgt,
        },
        lambda src, tgt: {
            "text": text,
            "source": src,
            "target": tgt,
        },
    ]

    last_error = None
    for src in source_candidates:
        for tgt in target_candidates:
            for build_payload in payload_templates:
                payload = build_payload(src, tgt)
                try:
                    response = requests.post(SARVAM_API_URL, json=payload, headers=headers, timeout=12)
                    response.raise_for_status()
                    translated = _extract_translated_text(response.json(), text)
                    if translated and translated.strip():
                        return translated
                except requests.HTTPError as exc:
                    err_body = ""
                    try:
                        err_body = exc.response.text
                    except Exception:
                        err_body = ""
                    last_error = f"{exc} | body={err_body[:400]}"
                except Exception as exc:
                    last_error = str(exc)

    print(
        f"[DEBUG][RAG] Sarvam translation failed ({source_lang}->{target_lang}). "
        f"All payload variants exhausted. Last error: {last_error}"
    )
    return text


def translate_to_english(text: str, source_lang: str) -> str:
    source_lang = normalize_lang_code(source_lang)
    return _translate_with_sarvam(text, source_lang, "en")


def translate_to_user_lang(text: str, target_lang: str) -> str:
    target_lang = normalize_lang_code(target_lang)
    return _translate_with_sarvam(text, "en", target_lang)


def retrieve_context(query: str, k: int = TOP_K):
    embedding = get_query_embedding(query)
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT_NAME,
        index_name=VECTOR_INDEX_NAME,
    )
    response = index.similarity_search(
        query_vector=embedding,
        columns=["chunk_text", "source_file", "page_number"],
        num_results=k,
    )
    rows = response.get("result", {}).get("data_array", [])
    print(f"[DEBUG][RAG] Retrieved rows: {len(rows)}")
    if rows:
        preview = [
            {
                "source_file": r[1] if len(r) > 1 else None,
                "page_number": r[2] if len(r) > 2 else None,
                "score": r[-1] if len(r) > 0 else None,
                "chunk_preview": (r[0][:180] + "...") if len(r) > 0 and r[0] and len(r[0]) > 180 else (r[0] if len(r) > 0 else ""),
            }
            for r in rows[:3]
        ]
        print(f"[DEBUG][RAG] Top retrieval preview: {preview}")
    return rows


def build_prompt(question: str, rows):
    context_lines = []
    for row in rows:
        chunk_text = row[0] if len(row) > 0 else ""
        source_file = row[1] if len(row) > 1 else "unknown"
        page_number = row[2] if len(row) > 2 else None
        context_lines.append(
            f"[Source: {source_file}, Page: {page_number}] {chunk_text}"
        )
    context_block = "\n\n".join(context_lines)[:MAX_CONTEXT_CHARS]
    return (
        "You are an Ayurvedic health assistant.\n"
        "Use the retrieved context as your first and strongest source of truth.\n"
        "If the context is weak, incomplete, or partially relevant, provide the best possible answer using reliable general Ayurvedic knowledge.\n"
        "When you use knowledge not explicitly present in retrieved context, clearly label it as general knowledge.\n"
        "Avoid hallucinations and avoid making diagnosis claims.\n\n"
        "Retrieved context:\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "Respond in 4-8 concise bullet points and include practical guidance where possible."
    )


def generate_answer(prompt: str):
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    w = WorkspaceClient()
    resp = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=512,
        temperature=0.2,
    )
    if resp.choices and len(resp.choices) > 0:
        return resp.choices[0].message.content
    raise ValueError("LLM returned no choices.")


def ask_multilingual(question: str, top_k: int = TOP_K, user_lang: Optional[str] = None) -> Dict:
    effective_top_k = max(3, min(int(top_k or TOP_K), 5))
    detected_lang = normalize_lang_code(user_lang) if user_lang else detect_language(question)
    if detected_lang not in SUPPORTED_LANGS:
        detected_lang = "en"

    english_question = question if detected_lang == "en" else translate_to_english(question, detected_lang)
    retrieval_rows = retrieve_context(english_question, effective_top_k)
    if not retrieval_rows:
        raise ValueError("No context retrieved from vector index.")

    prompt = build_prompt(english_question, retrieval_rows)
    english_answer = generate_answer(prompt)
    final_answer = english_answer if detected_lang == "en" else translate_to_user_lang(english_answer, detected_lang)

    return {
        "answer": final_answer,
        "answer_english": english_answer,
        "language": detected_lang,
        "sources": retrieval_rows,
        "question_english": english_question,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute RAG pipeline (optional — run this cell in the notebook only)

# COMMAND ----------

if __name__ == "__main__":
    try:
        result = ask_multilingual(USER_QUERY, TOP_K)
    except Exception as exc:
        raise RuntimeError("Multilingual RAG execution failed.") from exc

    print("Question:")
    print(USER_QUERY)
    print("\nDetected Language:")
    print(result["language"])
    print("\nAnswer:")
    print(result["answer"])

# COMMAND ----------

