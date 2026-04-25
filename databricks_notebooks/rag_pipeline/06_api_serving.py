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

# MAGIC %pip install fastapi uvicorn databricks-sdk databricks-vectorsearch requests

# COMMAND ----------

import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
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

app = FastAPI(title="AyurGenix RAG API", version="1.0.0")


class UserProfile(BaseModel):
    age: Optional[str] = None
    gender: Optional[str] = None
    lifestyle: Optional[str] = None
    stress_level: Optional[str] = None
    diet: Optional[str] = None
    sleep_quality: Optional[str] = None
    physical_activity: Optional[str] = None
    dietary_preference: Optional[str] = None
    health_goal: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=20)
    language: Optional[str] = Field(default=None, description="Preferred language code or name")
    user_profile: Optional[UserProfile] = None


class AskResponse(BaseModel):
    answer: str
    language: str
    sources: List[Dict[str, Any]]
    personalized_tips: List[str]
    latency_ms: int


def get_query_embedding(question: str):
    safe_query = question.replace("'", "\\'")
    row = spark.sql(
        f"SELECT ai_query('{EMBEDDING_ENDPOINT}', '{safe_query}') AS embedding"
    ).first()
    embedding = row["embedding"] if row else None
    if not embedding:
        raise ValueError("Failed to generate query embedding.")
    return embedding


def normalize_lang_code(code: Optional[str]) -> str:
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


def _translate_with_sarvam(text: str, source_lang: str, target_lang: str) -> str:
    if not text:
        return text
    if source_lang == target_lang:
        return text
    if not SARVAM_API_KEY:
        print("[DEBUG][API] SARVAM_API_KEY missing. Translation fallback returns original text.")
        return text

    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "mode": "formal",
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
    }
    try:
        response = requests.post(SARVAM_API_URL, json=payload, headers=headers, timeout=12)
        response.raise_for_status()
        data = response.json()
        return data.get("translated_text") or data.get("output") or data.get("translation") or text
    except Exception as exc:
        print(f"[DEBUG][API] Translation failed ({source_lang}->{target_lang}): {exc}")
        return text


def translate_to_english(text: str, source_lang: str) -> str:
    source_lang = normalize_lang_code(source_lang)
    return _translate_with_sarvam(text, source_lang, "en")


def translate_to_user_lang(text: str, target_lang: str) -> str:
    target_lang = normalize_lang_code(target_lang)
    return _translate_with_sarvam(text, "en", target_lang)


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
    print(f"[DEBUG][API] Retrieved {len(out)} contexts for question: {question}")
    if out:
        print(f"[DEBUG][API] Top context scores: {[c.get('score') for c in out[:5]]}")
    return out


def build_personalized_tips(user_profile: Optional[UserProfile], answer_text: str) -> List[str]:
    if not user_profile:
        return []

    tips: List[str] = []
    stress = (user_profile.stress_level or "").strip().lower()
    lifestyle = (user_profile.lifestyle or "").strip().lower()
    diet_pref = (user_profile.dietary_preference or user_profile.diet or "").strip().lower()
    sleep_quality = (user_profile.sleep_quality or "").strip().lower()
    activity = (user_profile.physical_activity or "").strip().lower()
    goal = (user_profile.health_goal or "").strip().lower()

    if stress == "high":
        tips.append("Stress is high: add 10-15 minutes of daily pranayama and evening digital detox.")
    if sleep_quality in {"poor", "low"}:
        tips.append("Sleep quality is low: prefer warm, light dinner and fixed bedtime before 10:30 PM.")
    if "sedentary" in lifestyle or activity in {"low", "minimal"}:
        tips.append("Low activity detected: include 30 minutes brisk walk or gentle yoga 5 days a week.")
    if "vegetarian" in diet_pref:
        tips.append("For vegetarian preference: include moong dal, ghee in moderation, and seasonal cooked vegetables.")
    if "digestion" in goal:
        tips.append("For digestion goal: use cumin-fennel-ginger infused warm water after meals.")
    if "joint" in goal:
        tips.append("For joint care: add mobility stretches and anti-inflammatory spices like turmeric.")
    if "stress" in answer_text.lower() and not any("stress" in t.lower() for t in tips):
        tips.append("Prioritize stress reduction because it can worsen multiple Ayurvedic imbalance patterns.")

    return tips[:5]


def build_prompt(question: str, contexts: List[Dict[str, Any]], user_profile: Optional[UserProfile]) -> str:
    context_block = "\n\n".join(
        [
            f"[Source: {c.get('source_file')}, Page: {c.get('page_number')}] {c.get('chunk_text')}"
            for c in contexts
        ]
    )[:MAX_CONTEXT_CHARS]
    profile_block = (
        f"Age: {user_profile.age or 'unknown'}\n"
        f"Gender: {user_profile.gender or 'unknown'}\n"
        f"Lifestyle: {user_profile.lifestyle or 'unknown'}\n"
        f"Stress Level: {user_profile.stress_level or 'unknown'}\n"
        f"Diet: {user_profile.dietary_preference or user_profile.diet or 'unknown'}\n"
        f"Sleep Quality: {user_profile.sleep_quality or 'unknown'}\n"
        f"Physical Activity: {user_profile.physical_activity or 'unknown'}\n"
        f"Health Goal: {user_profile.health_goal or 'unknown'}"
        if user_profile
        else "Not provided"
    )
    return (
        "You are an Ayurvedic health assistant.\n"
        "Prioritize retrieved context when answering.\n"
        "If retrieved context is insufficient, provide a best-effort answer using reliable general Ayurvedic knowledge and explicitly mark that portion as general knowledge.\n"
        "Do not invent citations; only cite the provided sources.\n\n"
        f"User profile for personalization (light touch):\n{profile_block}\n\n"
        "Retrieved context:\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n"
        "Answer clearly and include concise practical guidance."
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


def _profile_cache_key(profile: Optional[UserProfile]) -> Tuple[str, ...]:
    if not profile:
        return tuple()
    return (
        str(profile.age or ""),
        str(profile.gender or ""),
        str(profile.lifestyle or ""),
        str(profile.stress_level or ""),
        str(profile.diet or ""),
        str(profile.sleep_quality or ""),
        str(profile.physical_activity or ""),
        str(profile.dietary_preference or ""),
        str(profile.health_goal or ""),
    )


@lru_cache(maxsize=256)
def _cached_english_answer(question_en: str, top_k: int, profile_key: Tuple[str, ...]) -> Tuple[str, Tuple[Tuple[Any, ...], ...]]:
    contexts = retrieve(question_en, top_k)
    if not contexts:
        raise ValueError("No relevant chunks retrieved.")

    profile_for_prompt = UserProfile(
        age=profile_key[0] if len(profile_key) > 0 else None,
        gender=profile_key[1] if len(profile_key) > 1 else None,
        lifestyle=profile_key[2] if len(profile_key) > 2 else None,
        stress_level=profile_key[3] if len(profile_key) > 3 else None,
        diet=profile_key[4] if len(profile_key) > 4 else None,
        sleep_quality=profile_key[5] if len(profile_key) > 5 else None,
        physical_activity=profile_key[6] if len(profile_key) > 6 else None,
        dietary_preference=profile_key[7] if len(profile_key) > 7 else None,
        health_goal=profile_key[8] if len(profile_key) > 8 else None,
    ) if profile_key else None

    prompt = build_prompt(question_en, contexts, profile_for_prompt)
    answer_en = call_llm(prompt)
    context_tuple = tuple(
        (
            c.get("chunk_text"),
            c.get("source_file"),
            c.get("page_number"),
            c.get("score"),
        )
        for c in contexts
    )
    return answer_en, context_tuple


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    started_at = time.time()
    try:
        target_lang = normalize_lang_code(req.language) if req.language else detect_language(req.question)
        if target_lang not in SUPPORTED_LANGS:
            target_lang = "en"

        question_en = req.question if target_lang == "en" else translate_to_english(req.question, target_lang)
        effective_top_k = max(3, min(int(req.top_k), 5))

        answer_en, context_tuple = _cached_english_answer(
            question_en,
            effective_top_k,
            _profile_cache_key(req.user_profile),
        )
        contexts = [
            {
                "chunk_text": row[0],
                "source_file": row[1],
                "page_number": row[2],
                "score": row[3],
            }
            for row in context_tuple
        ]

        answer = answer_en if target_lang == "en" else translate_to_user_lang(answer_en, target_lang)
        tips = build_personalized_tips(req.user_profile, answer_en)
        latency_ms = int((time.time() - started_at) * 1000)

        return AskResponse(
            answer=answer,
            language=target_lang,
            sources=contexts,
            personalized_tips=tips,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run locally in Databricks driver (for testing)

# COMMAND ----------

# MAGIC %sh
# MAGIC # uvicorn main:app --host 0.0.0.0 --port 8000

# COMMAND ----------


