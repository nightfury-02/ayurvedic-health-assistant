import streamlit as st
from backend.test_match import predict

st.set_page_config(page_title="Ayurveda AI", layout="centered")

st.title("🌿 Ayurveda AI Assistant")

user_input = st.text_input("Enter symptoms (e.g., cough, throat pain)")

if st.button("Analyze"):
    result = predict(user_input)

    st.subheader("🧠 Analysis Result")

    st.write(f"**Disease:** {result['disease']}")

    st.write("**Dosha Imbalance:**")
    for d in result["dosha"]:
        st.write(f"• {d} ↑")

    st.write("**Recommendation:**")
    st.info(result["advice"])