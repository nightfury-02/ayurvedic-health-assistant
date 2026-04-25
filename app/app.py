import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# Support `streamlit run app/app.py` (repo root cwd) and `streamlit run app.py` (app/ cwd).
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from databricks_rag import (
    ask_via_http,
    get_ask_multilingual,
    normalize_sources,
    resolve_backend_url,
)


@st.cache_resource
def _cached_ask_multilingual():
    """Load RAG once per process (Databricks driver / App worker)."""
    return get_ask_multilingual()


def ask_assistant(
    question: str,
    top_k: int,
    user_lang: str,
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    backend = resolve_backend_url()
    if backend:
        return ask_via_http(backend, question, top_k, user_lang, user_profile)

    ask_fn = _cached_ask_multilingual()
    # Notebook RAG (05_rag_pipeline) personalizes via language + retrieval only; profile is used in the UI sidebar.
    data = ask_fn(
        question=question,
        top_k=top_k,
        user_lang=user_lang,
    )
    return {
        "answer": data.get("answer", ""),
        "sources": normalize_sources(data.get("sources")),
        "personalized_tips": [],
    }


# ================= LANGUAGE =================
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
}


# ================= HELPERS =================
def extract_health_insights(answer: str, profile: Dict[str, Any]) -> Dict[str, str]:
    answer_l = (answer or "").lower()
    goal = (profile.get("health_goal") or "").strip() or "general wellness"
    stress = (profile.get("stress_level") or "").strip().lower()

    issue = "General wellness imbalance"
    if "sleep" in answer_l:
        issue = "Sleep quality concern"
    elif "digestion" in answer_l or "gut" in answer_l:
        issue = "Digestive imbalance"
    elif "joint" in answer_l or "pain" in answer_l:
        issue = "Joint discomfort pattern"
    elif stress == "high":
        issue = "Stress-driven lifestyle imbalance"

    improvement = f"Focus on improving {goal} with consistent daily habits."
    return {"issue": issue, "improvement": improvement}


def highlight_keywords(answer: str) -> str:
    keywords = ["diet", "sleep", "stress", "yoga", "exercise", "dosha", "herb", "ayurveda"]
    highlighted = answer or ""
    for word in keywords:
        highlighted = re.sub(rf"\b({word})\b", r"**\1**", highlighted, flags=re.IGNORECASE)
    return highlighted


def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ================= UI =================
st.set_page_config(page_title="AyurGenix Assistant", page_icon="🌿", layout="wide")
init_state()

st.title("🌿 AyurGenix Multilingual Health Assistant")
_backend = resolve_backend_url()
if _backend:
    st.caption(f"Databricks: Streamlit → RAG API at `{_backend}` (`AYURGENIX_BACKEND_URL`).")
else:
    st.caption(
        "Databricks: in-process multilingual RAG (vector search + LLM on this workspace). "
        "Optional: set `AYURGENIX_BACKEND_URL` to a served `/ask` endpoint instead."
    )

# ================= SIDEBAR =================
with st.sidebar:
    st.header("User Profile")

    age = st.text_input("Age")
    gender = st.selectbox("Gender", ["", "Female", "Male", "Other"])
    stress_level = st.selectbox("Stress Level", ["", "Low", "Medium", "High"])
    sleep_quality = st.selectbox("Sleep Quality", ["", "Poor", "Average", "Good"])
    physical_activity = st.selectbox("Physical Activity", ["", "Low", "Moderate", "High"])
    dietary_preference = st.selectbox("Dietary Preference", ["", "Vegetarian", "Vegan", "Mixed"])
    lifestyle = st.selectbox("Lifestyle", ["", "Sedentary", "Moderately active", "Active"])
    health_goal = st.text_input("Health Goal")

    selected_lang_label = st.selectbox("Response Language", list(LANGUAGE_OPTIONS.keys()))
    top_k = st.slider("Retrieval depth", 3, 5, 4)

    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()


profile_payload = {
    "age": age,
    "gender": gender,
    "lifestyle": lifestyle,
    "stress_level": stress_level,
    "diet": dietary_preference,
    "sleep_quality": sleep_quality,
    "physical_activity": physical_activity,
    "health_goal": health_goal,
}


# ================= LAYOUT =================
chat_col, dashboard_col = st.columns([2, 1])


# ================= CHAT =================
with chat_col:
    st.subheader("Chat")

    question = st.text_input(
        "Ask your question",
        placeholder="How can I manage diabetes with Ayurveda?",
    )

    if st.button("Ask Assistant") and question.strip():
        try:
            with st.spinner("🔍 Retrieving knowledge + generating answer..."):
                data = ask_assistant(
                    question=question.strip(),
                    top_k=top_k,
                    user_lang=LANGUAGE_OPTIONS[selected_lang_label],
                    user_profile=profile_payload,
                )

            st.session_state.chat_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": question,
                    "answer": data.get("answer", ""),
                    "sources": data.get("sources", []),
                    "tips": data.get("personalized_tips", []),
                    "profile": profile_payload,
                }
            )

        except requests.RequestException as e:
            st.error(
                f"❌ Request to RAG API failed ({_backend}). "
                f"Check `AYURGENIX_BACKEND_URL` and that the service exposes POST /ask. Details: {e}"
            )
        except FileNotFoundError as e:
            st.error(
                f"❌ {e} "
                "Deploy the repository so `databricks_notebooks/rag_pipeline/05_rag_pipeline.py` exists, "
                "or set `AYURGENIX_REPO_ROOT` to that checkout path."
            )
        except Exception as e:
            st.error(f"❌ RAG failed: {e}")


    # ================= SHOW CHAT =================
    for item in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {highlight_keywords(item['answer'])}")

            with st.expander("Sources"):
                for src in item.get("sources", []):
                    st.write(f"📄 {src.get('source_file')}")
                    preview = (src.get("chunk_text") or "")[:300]
                    st.caption(preview + ("..." if len(src.get("chunk_text") or "") > 300 else ""))


# ================= DASHBOARD =================
with dashboard_col:
    st.subheader("📊 Insights")

    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]

        insights = extract_health_insights(
            latest.get("answer", ""),
            latest.get("profile", {}),
        )

        st.write("**Issue:**", insights["issue"])
        st.write("**Improvement:**", insights["improvement"])

        st.markdown("### 🌿 Suggestions")
        tips = latest.get("tips") or []
        if tips:
            for tip in tips:
                st.write(f"- {tip}")
        else:
            st.write("- Maintain daily routine (dinacharya)")
            st.write("- Follow balanced Ayurvedic diet")
            st.write("- Practice yoga / meditation")

    else:
        st.info("Ask a question to see insights")
