"""Streamlit frontend for the AyurGenix RAG assistant.

Two backend modes:

* **Direct** — calls the in-process Python RAG pipeline (``backend.rag_core``).
  This is the recommended mode when Streamlit and Databricks share credentials
  (Databricks Apps deployment, or local dev with ``DATABRICKS_HOST`` /
  ``DATABRICKS_TOKEN`` env vars set). No HTTP, no separate API server.

* **Remote API** — calls a stand-alone FastAPI service (the one defined in
  ``databricks_notebooks/rag_pipeline/06_api_serving.py``) over HTTP. Use this
  when the RAG service runs on a different host than Streamlit.

Two tabs:

1. **Ask AyurGenix** — multilingual RAG (any supported regional language).
2. **Quick demo (offline)** — original rule-based matcher; works without any
   Databricks connectivity.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import streamlit as st

from backend.api_client import AyurGenixAPIError, AyurGenixClient
from backend.test_match import predict


# -- Defaults from environment ------------------------------------------------
DEFAULT_API_URL = os.getenv("AYURGENIX_API_URL", "")
DEFAULT_API_TOKEN = os.getenv("AYURGENIX_API_TOKEN", "")

# Static fallback — used if rag_core can't import (no SDK installed yet) AND
# no remote API is reachable. Mirrors rag_core.SUPPORTED_LANGUAGES.
FALLBACK_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "or": "Odia",
    "ur": "Urdu",
}

EXAMPLE_QUERIES = {
    "English":  "How does Ayurveda recommend improving sleep quality?",
    "Hindi":    "अच्छी नींद के लिए आयुर्वेद क्या सुझाता है?",
    "Telugu":   "మంచి నిద్ర కోసం ఆయుర్వేదం ఏమి సూచిస్తుంది?",
    "Tamil":    "நல்ல தூக்கத்திற்கு ஆயுர்வேதம் என்ன பரிந்துரைக்கிறது?",
    "Kannada":  "ಒಳ್ಳೆಯ ನಿದ್ರೆಗಾಗಿ ಆಯುರ್ವೇದ ಏನು ಶಿಫಾರಸು ಮಾಡುತ್ತದೆ?",
    "Bengali":  "ভালো ঘুমের জন্য আয়ুর্বেদ কী পরামর্শ দেয়?",
}


# -- Try to import the in-process RAG core. If it fails (e.g. databricks-sdk
# not installed), Direct mode will be disabled and the app falls back to
# Remote API mode automatically.
try:
    from backend import rag_core  # type: ignore
    DIRECT_AVAILABLE = True
    DIRECT_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001
    rag_core = None  # type: ignore
    DIRECT_AVAILABLE = False
    DIRECT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _html_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _render_answer(
    answer: str,
    detected_code: str,
    language_code: str,
    languages_map: dict,
    disclaimer: str,
    sources: List[Any],
) -> None:
    detected_name = languages_map.get(detected_code, detected_code)
    response_name = languages_map.get(language_code, language_code)
    st.caption(
        f"Detected query language: **{detected_name}** · "
        f"Answer in: **{response_name}**"
    )
    st.subheader("Answer")
    st.markdown(
        f"<div style='font-size:1.05rem; line-height:1.6;'>"
        f"{_html_escape(answer)}</div>",
        unsafe_allow_html=True,
    )
    if disclaimer:
        st.info(disclaimer)
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for i, src in enumerate(sources, start=1):
                # Works for both rag_core.Source and api_client.Source dataclasses.
                source_file = getattr(src, "source_file", None)
                page_number = getattr(src, "page_number", None)
                score = getattr(src, "score", None)
                chunk_text = getattr(src, "chunk_text", "") or ""
                page_str = f" · p.{page_number}" if page_number is not None else ""
                score_str = f" · score {score:.3f}" if isinstance(score, float) else ""
                st.markdown(f"**{i}. {source_file or 'unknown'}**{page_str}{score_str}")
                snippet = chunk_text.strip()
                if len(snippet) > 600:
                    snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
                st.caption(snippet)
                st.divider()
    else:
        st.caption("No supporting sources were returned.")


# ---- Page setup -------------------------------------------------------------

st.set_page_config(
    page_title="AyurGenix Assistant",
    page_icon="🌿",
    layout="wide",
)


# ---- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("AyurGenix")
    st.caption(
        "Multilingual Ayurveda Q&A grounded in a Databricks-hosted knowledge "
        "base (Vector Search + Llama 3.3)."
    )
    st.divider()

    st.subheader("Backend")
    mode_options = []
    if DIRECT_AVAILABLE:
        mode_options.append("Direct (in-process)")
    mode_options.append("Remote API (HTTP)")
    mode = st.radio(
        "Mode",
        mode_options,
        index=0,
        help=(
            "Direct: calls the Databricks Vector Search + LLM endpoints "
            "in-process. Best for Databricks Apps deployments. "
            "Remote API: hits the FastAPI service from notebook 06."
        ),
    )

    if not DIRECT_AVAILABLE:
        st.warning(
            "Direct mode unavailable: could not import `backend.rag_core`. "
            f"Reason: {DIRECT_IMPORT_ERROR}. Install the dependencies in "
            "`requirements.txt` to enable it."
        )

    using_direct = mode.startswith("Direct")

    if not using_direct:
        st.subheader("Remote API")
        api_url = st.text_input(
            "API URL",
            value=DEFAULT_API_URL,
            placeholder="https://my-fastapi-host.example.com",
            help=(
                "Base URL of the FastAPI service from notebook 06. "
                "Include scheme (http:// or https://) and port if non-default."
            ),
        )
        api_token = st.text_input(
            "Bearer token (optional)",
            value=DEFAULT_API_TOKEN,
            type="password",
            help=(
                "Databricks PAT or service-principal token if the API "
                "requires authentication."
            ),
        )
        if st.button("Check connection", use_container_width=True):
            try:
                health = AyurGenixClient(api_url, token=api_token or None).health()
                st.success(f"OK · {health}")
            except AyurGenixAPIError as exc:
                detail = f" ({exc.detail})" if exc.detail else ""
                st.error(f"Cannot reach API{detail}: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")
    else:
        api_url = ""
        api_token = ""
        with st.expander("Direct mode config", expanded=False):
            st.caption(
                "These constants live in `backend/rag_core.py` and can be "
                "overridden via env vars: `AYURGENIX_CATALOG`, "
                "`AYURGENIX_SCHEMA`, `AYURGENIX_VECTOR_ENDPOINT`, "
                "`AYURGENIX_VECTOR_INDEX`, `AYURGENIX_EMBEDDING_ENDPOINT`, "
                "`AYURGENIX_LLM_ENDPOINT`."
            )
            st.code(
                f"Catalog       : {rag_core.CATALOG}\n"
                f"Schema        : {rag_core.SCHEMA}\n"
                f"Endpoint      : {rag_core.VECTOR_ENDPOINT_NAME}\n"
                f"Vector index  : {rag_core.VECTOR_INDEX_NAME}\n"
                f"Embedding     : {rag_core.EMBEDDING_ENDPOINT}\n"
                f"LLM           : {rag_core.LLM_ENDPOINT}",
                language="text",
            )

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Top-K passages", min_value=1, max_value=10, value=5)
    source_filter_label = st.selectbox(
        "Source filter",
        options=["All", "CSV (herb monographs)", "PDF (textbooks)"],
        index=0,
    )
    source_filter: Optional[str] = (
        "csv" if "CSV" in source_filter_label
        else "pdf" if "PDF" in source_filter_label
        else None
    )

    st.divider()
    st.caption(
        "Educational use only. Not a substitute for professional medical "
        "advice. Always consult a qualified clinician for health decisions."
    )


# ---- Main: tabs --------------------------------------------------------------

st.title("🌿 AyurGenix Assistant")

tab_rag, tab_demo = st.tabs(["Ask AyurGenix (live RAG)", "Quick demo (offline)"])


# ---- Tab 1: Live RAG --------------------------------------------------------

with tab_rag:
    st.markdown(
        "Ask in **English** or any supported regional language "
        "(Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Punjabi, "
        "Gujarati, Odia, Urdu). The answer comes back in the same language "
        "you asked in — or pick a different one below."
    )

    # Build the language map: prefer rag_core.SUPPORTED_LANGUAGES in Direct
    # mode; otherwise fetch from the API; otherwise fall back to the static map.
    if using_direct and DIRECT_AVAILABLE:
        languages = {"auto": "Auto-detect", **rag_core.SUPPORTED_LANGUAGES}
    else:
        if "languages_cache" not in st.session_state:
            try:
                api_langs = (
                    AyurGenixClient(api_url, token=api_token or None).languages()
                )
                st.session_state["languages_cache"] = {"auto": "Auto-detect", **api_langs}
            except Exception:  # noqa: BLE001
                st.session_state["languages_cache"] = FALLBACK_LANGUAGES
        languages = st.session_state["languages_cache"]

    col_q, col_lang = st.columns([3, 1])
    with col_lang:
        lang_label = st.selectbox(
            "Response language",
            options=list(languages.values()),
            index=0,
            help="Auto-detect uses the script of your question.",
        )
        lang_code: Optional[str] = next(
            (code for code, name in languages.items() if name == lang_label),
            None,
        )
        if lang_code == "auto":
            lang_code = None

    with col_q:
        st.caption("Try an example:")
        ex_cols = st.columns(min(len(EXAMPLE_QUERIES), 6))
        for col, (lang_name, q) in zip(ex_cols, EXAMPLE_QUERIES.items()):
            if col.button(lang_name, use_container_width=True, key=f"ex_{lang_name}"):
                st.session_state["rag_query"] = q

        question = st.text_area(
            "Your question",
            value=st.session_state.get("rag_query", ""),
            key="rag_query",
            height=120,
            placeholder="e.g. How does Ayurveda recommend improving sleep quality?",
        )

    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Querying AyurGenix knowledge base…"):
            try:
                if using_direct:
                    if not DIRECT_AVAILABLE:
                        st.error("Direct mode is unavailable; switch to Remote API.")
                        st.stop()
                    result = rag_core.ask(
                        question=question,
                        top_k=top_k,
                        source_filter=source_filter,
                        language=lang_code,
                    )
                else:
                    if not api_url:
                        st.error(
                            "Remote API mode selected but no API URL configured. "
                            "Set one in the sidebar."
                        )
                        st.stop()
                    client = AyurGenixClient(api_url, token=api_token or None)
                    result = client.ask(
                        question=question,
                        top_k=top_k,
                        source_filter=source_filter,
                        language=lang_code,
                    )
            except AyurGenixAPIError as exc:
                detail = f" — {exc.detail}" if exc.detail else ""
                st.error(f"AyurGenix API error{detail}\n\n{exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"RAG pipeline failed: {type(exc).__name__}: {exc}")
                st.stop()

        _render_answer(
            answer=result.answer,
            detected_code=result.detected_language,
            language_code=result.language,
            languages_map=languages,
            disclaimer=result.disclaimer,
            sources=list(result.sources or []),
        )


# ---- Tab 2: Offline demo ----------------------------------------------------

with tab_demo:
    st.markdown(
        "Rule-based symptom matcher (English only). Useful when the "
        "Databricks service is offline or for quick screenshots."
    )
    examples = [
        "cough and throat pain",
        "fatigue, frequent urination, thirst",
        "high bp and stress headaches",
    ]
    example_choice = st.radio("Try an example", examples, index=None, horizontal=True)
    user_input = st.text_input(
        "Symptoms",
        value=example_choice or st.session_state.get("demo_input", ""),
        key="demo_input",
        placeholder="cough, throat pain, congestion",
    )

    if st.button("Analyze", type="primary", key="demo_analyze"):
        if not user_input.strip():
            st.warning("Please enter at least one symptom.")
            st.stop()
        try:
            result = predict(user_input)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not analyze input: {exc}")
            st.stop()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.metric("Likely condition", result["disease"])
        with col_b:
            confidence = result.get("confidence", 0.0)
            st.metric("Match confidence", f"{int(confidence * 100)}%")

        matched = result.get("matched_keywords") or []
        if matched:
            st.caption("Matched keywords: " + ", ".join(f"`{k}`" for k in matched))

        st.write("**Dosha context:**")
        dosha_cols = st.columns(max(len(result["dosha"]), 1))
        for col, dosha in zip(dosha_cols, result["dosha"]):
            col.success(f"{dosha} ↑")

        st.write("**Recommendation:**")
        st.info(result["advice"])
