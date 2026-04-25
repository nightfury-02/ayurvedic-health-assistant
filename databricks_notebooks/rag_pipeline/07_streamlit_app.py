# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Streamlit App (Optional UI)
# MAGIC
# MAGIC Simple chat UI that calls the FastAPI `/ask` endpoint.
# MAGIC
# MAGIC Update `API_BASE_URL` to your deployed API URL.

# COMMAND ----------

# MAGIC %pip install streamlit requests

# COMMAND ----------

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"

st.set_page_config(page_title="AyurGenix Assistant", page_icon="🌿", layout="wide")
st.title("🌿 AyurGenix RAG Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a health question", placeholder="e.g. How does Ayurveda improve gut health?")
top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=5, step=1)

if st.button("Ask") and question.strip():
    try:
        payload = {"question": question.strip(), "top_k": int(top_k)}
        resp = requests.post(ASK_ENDPOINT, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.history.append(
            {
                "question": question.strip(),
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
            }
        )
    except Exception as exc:
        st.error(f"Request failed: {exc}")

for item in reversed(st.session_state.history):
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")
    with st.expander("Sources"):
        for src in item["sources"]:
            st.write(
                {
                    "source_file": src.get("source_file"),
                    "page_number": src.get("page_number"),
                    "score": src.get("score"),
                }
            )
            st.caption(src.get("chunk_text", "")[:800] + "...")
