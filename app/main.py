%pip install streamlit
import streamlit as st

st.set_page_config(page_title="Ayurveda AI", layout="centered")

st.title("Ayurveda AI Assistant")

user_input = st.text_input("Enter symptoms")

def analyze(text: str):
    if not text:
        return "Please enter symptoms."
    # dummy logic
    if "cough" in text.lower():
        return "Possible condition: Cough\nDosha: Kapha ↑"
    return "Possible condition: General imbalance\nDosha: Vata ↑"

if st.button("Analyze"):
    result = analyze(user_input)
    st.write(result)