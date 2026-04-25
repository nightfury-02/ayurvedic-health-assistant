from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="AyurGenix Local Backend", version="1.0.0")


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
    top_k: int = Field(default=4, ge=1, le=20)
    language: Optional[str] = "en"
    user_profile: Optional[UserProfile] = None


class AskResponse(BaseModel):
    answer: str
    language: str
    sources: List[Dict[str, Any]]
    personalized_tips: List[str]
    latency_ms: int


def build_tips(profile: Optional[UserProfile]) -> List[str]:
    if not profile:
        return [
            "Maintain consistent sleep and meal timing.",
            "Prefer warm, freshly prepared meals for easier digestion.",
        ]

    tips: List[str] = []
    stress = (profile.stress_level or "").lower()
    goal = (profile.health_goal or "").lower()
    lifestyle = (profile.lifestyle or "").lower()

    if stress == "high":
        tips.append("Practice 10 minutes of pranayama daily to reduce stress.")
    if "digestion" in goal:
        tips.append("Use warm water with cumin-fennel after meals.")
    if "joint" in goal:
        tips.append("Add gentle mobility exercises and turmeric in food.")
    if "sedentary" in lifestyle:
        tips.append("Take a 20-30 minute walk daily to improve circulation.")

    return tips or [
        "Follow a regular daily routine (dinacharya).",
        "Balance meals with seasonal foods and good hydration.",
    ]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    tips = build_tips(req.user_profile)
    answer = (
        f"Here is a practical Ayurvedic response to your question: '{req.question}'. "
        "Start with consistent sleep, balanced meals, and stress management. "
        "For clinical concerns, consult a qualified practitioner."
    )
    sources = [
        {
            "source_file": "local-backend",
            "page_number": None,
            "score": 1.0,
            "chunk_text": "Local fallback backend response (Databricks RAG not connected).",
        }
    ]

    return AskResponse(
        answer=answer,
        language=req.language or "en",
        sources=sources,
        personalized_tips=tips,
        latency_ms=30,
    )
