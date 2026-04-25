import os
import streamlit as st

from backend.config import (
    CURATED_TABLE,
    SQL_WAREHOUSE_CONFIGURED,
    USE_LOCAL_FALLBACK,
    VECTOR_SEARCH_ENDPOINT,
    VECTOR_SEARCH_INDEX_CURATED,
    VECTOR_SEARCH_INDEX_PDF,
)
from backend.rag import legacy_symptom_card, run_chat

st.set_page_config(page_title="Ayurveda AI", layout="wide")

st.title("Ayurveda AI Assistant")
st.caption("Hybrid RAG: Mosaic AI Vector Search + Delta SQL + Sarvam (educational use only, not medical advice).")

with st.sidebar:
    st.subheader("Runtime")
    st.write("**Lakehouse mode:**", "off (local CSV)" if USE_LOCAL_FALLBACK else "on (Databricks)")
    st.write("**SQL warehouse (AyurGenix SQL):**", "yes" if SQL_WAREHOUSE_CONFIGURED else "no")
    st.write("**Curated table:**", CURATED_TABLE)
    st.write("**PDF vector index:**", VECTOR_SEARCH_INDEX_PDF or "(not set)")
    st.write("**Curated vector index:**", VECTOR_SEARCH_INDEX_CURATED or "(not set)")
    st.write("**VS endpoint:**", VECTOR_SEARCH_ENDPOINT or "(not set)")
    if not os.getenv("SARVAM_API_KEY"):
        st.warning("Set `SARVAM_API_KEY` for full LLM answers.")

tab_chat, tab_symptoms = st.tabs(["Multilingual chat (RAG)", "Structured symptom card"])

with tab_chat:
    q = st.text_area(
        "Ask in English or an Indic language (symptoms, herbs, formulations, PDF concepts):",
        height=100,
        key="chat_q",
    )
    if st.button("Run RAG", key="btn_rag"):
        if not (q or "").strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving evidence and generating…"):
                out = run_chat(q)
            for warn in out.get("warnings") or []:
                st.warning(warn)
            st.markdown(out.get("answer") or "_Empty response_")
            with st.expander("Retrieved PDF chunks (metadata)"):
                st.json(out.get("chunk_hits") or [])
            with st.expander("Retrieved AyurGenix rows (SQL or local)"):
                st.json(out.get("curated_hits") or [])
            with st.expander("Retrieved AyurGenix rows (curated vector index)"):
                st.json(out.get("curated_vector_hits") or [])
            with st.expander("Context preview (truncated)"):
                st.text(out.get("context_preview") or "")

with tab_symptoms:
    user_input = st.text_input("Symptoms (e.g. cough, throat pain, खांसी)", key="sym_in")
    if st.button("Analyze", key="btn_sym"):
        with st.spinner("Running hybrid retrieval…"):
            result = legacy_symptom_card(user_input)
        for warn in result.get("warnings") or []:
            st.warning(warn)
        st.subheader("Structured match (best curated row)")
        st.write(f"**Disease:** {result.get('disease')}")
        st.write("**Dosha:**")
        for d in result.get("dosha") or []:
            st.write(f"• {d}")
        st.write("**Recommendation (dataset fields):**")
        st.info(result.get("advice") or "")
        if result.get("llm_answer"):
            st.subheader("Sarvam explanation")
            st.markdown(result["llm_answer"])
