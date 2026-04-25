import re
from datetime import datetime
from typing import Any, Dict, List

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"

LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
}


@st.cache_data(ttl=120)
def ask_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(ASK_ENDPOINT, json=payload, timeout=45)
    response.raise_for_status()
    return response.json()


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

    improvement = f"Prioritize a consistent routine focused on {goal} with daily small improvements."
    if "general knowledge" in answer_l:
        improvement = "Knowledge base coverage is partial for this query; combine guidance with clinician validation."

    return {"issue": issue, "improvement": improvement}


def highlight_keywords(answer: str) -> str:
    keywords = ["diet", "sleep", "stress", "yoga", "exercise", "dosha", "herb", "ayurveda", "digestion", "joint"]
    highlighted = answer or ""
    for word in keywords:
        highlighted = re.sub(
            rf"\b({word})\b",
            r"**\1**",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_history_index" not in st.session_state:
        st.session_state.selected_history_index = None


st.set_page_config(page_title="AyurGenix Multilingual Assistant", page_icon="🌿", layout="wide")
init_state()

st.title("🌿 AyurGenix Multilingual Health Assistant")
st.caption("Multilingual Ayurvedic RAG assistant with profile-aware recommendations.")

with st.sidebar:
    st.header("User Profile")
    age = st.text_input("Age", value="")
    gender = st.selectbox("Gender", ["", "Female", "Male", "Other"])
    stress_level = st.selectbox("Stress Level", ["", "Low", "Medium", "High"])
    sleep_quality = st.selectbox("Sleep Quality", ["", "Poor", "Average", "Good"])
    physical_activity = st.selectbox("Physical Activity", ["", "Low", "Moderate", "High"])
    dietary_preference = st.selectbox("Dietary Preference", ["", "Vegetarian", "Vegan", "Mixed", "Other"])
    lifestyle = st.selectbox("Lifestyle", ["", "Sedentary", "Moderately active", "Active"])
    health_goal = st.text_input("Health Goal", value="", placeholder="e.g., digestion, joint pain, stress")

    st.divider()
    selected_lang_label = st.selectbox("Response Language", list(LANGUAGE_OPTIONS.keys()), index=0)
    top_k = st.slider("Retrieval depth (top_k)", min_value=3, max_value=5, value=4, step=1)

    if st.button("Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.selected_history_index = None
        st.rerun()

profile_payload = {
    "age": age,
    "gender": gender,
    "lifestyle": lifestyle,
    "stress_level": stress_level,
    "diet": dietary_preference,
    "sleep_quality": sleep_quality,
    "physical_activity": physical_activity,
    "dietary_preference": dietary_preference,
    "health_goal": health_goal,
}

chat_col, dashboard_col = st.columns([2, 1], gap="large")

with chat_col:
    st.subheader("Chat")
    question = st.text_input(
        "Ask your question",
        placeholder="How can I improve digestion naturally with Ayurveda?",
        key="question_input",
    )
    submit = st.button("Ask Assistant", type="primary")

    if submit and question.strip():
        payload = {
            "question": question.strip(),
            "language": LANGUAGE_OPTIONS[selected_lang_label],
            "top_k": int(top_k),
            "user_profile": profile_payload,
        }
        try:
            with st.spinner("Thinking and retrieving relevant Ayurvedic context..."):
                data = ask_api(payload)
            st.session_state.chat_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": question.strip(),
                    "answer": data.get("answer", ""),
                    "language": data.get("language", "en"),
                    "sources": data.get("sources", []),
                    "personalized_tips": data.get("personalized_tips", []),
                    "latency_ms": data.get("latency_ms", None),
                    "profile": profile_payload,
                }
            )
        except Exception as exc:
            st.error(f"API request failed: {exc}")

    for idx, item in enumerate(reversed(st.session_state.chat_history)):
        actual_index = len(st.session_state.chat_history) - 1 - idx
        with st.container(border=True):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {highlight_keywords(item['answer'])}")
            st.caption(f"Language: `{item.get('language', 'en')}` | Latency: `{item.get('latency_ms', 'n/a')} ms`")
            if st.button("View on dashboard", key=f"view_{actual_index}"):
                st.session_state.selected_history_index = actual_index
            with st.expander("Sources"):
                for src in item.get("sources", []):
                    st.write(
                        {
                            "source_file": src.get("source_file"),
                            "page_number": src.get("page_number"),
                            "score": src.get("score"),
                        }
                    )
                    snippet = src.get("chunk_text", "") or ""
                    st.caption((snippet[:700] + "...") if len(snippet) > 700 else snippet)

with dashboard_col:
    st.subheader("Personalized Dashboard")
    selected = None
    if st.session_state.selected_history_index is not None and st.session_state.chat_history:
        if 0 <= st.session_state.selected_history_index < len(st.session_state.chat_history):
            selected = st.session_state.chat_history[st.session_state.selected_history_index]
    elif st.session_state.chat_history:
        selected = st.session_state.chat_history[-1]

    if not selected:
        st.info("Ask a question to populate personalized insights.")
    else:
        insights = extract_health_insights(selected.get("answer", ""), selected.get("profile", {}))
        st.markdown("#### 📊 Health Insights")
        st.write(f"**Key detected issue:** {insights['issue']}")
        st.write(f"**Suggested improvement:** {insights['improvement']}")

        st.markdown("#### 🥗 Diet Recommendations")
        tips = selected.get("personalized_tips", [])
        diet_related = [t for t in tips if "diet" in t.lower() or "vegetarian" in t.lower() or "meal" in t.lower()]
        for tip in (diet_related or tips[:2] or ["Prefer warm, freshly cooked meals and maintain consistent meal timing."]):
            st.write(f"- {tip}")

        st.markdown("#### 🧘 Lifestyle Suggestions")
        lifestyle_related = [t for t in tips if "stress" in t.lower() or "activity" in t.lower() or "yoga" in t.lower() or "sleep" in t.lower()]
        for tip in (lifestyle_related or tips[:2] or ["Maintain regular sleep and include daily light movement such as yoga or walking."]):
            st.write(f"- {tip}")

        st.markdown("#### 🌿 Ayurvedic Suggestions")
        st.write("- Favor routine (`dinacharya`) and season-aligned food choices.")
        st.write("- Use personalized herbs only with qualified practitioner guidance.")
        st.write("- Track symptom trends weekly and iterate habits gradually.")
