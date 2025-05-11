import streamlit as st
from datasets import load_dataset

# Load the FAQ dataset from Hugging Face
@st.cache_resource
def load_faq_dataset():
    dataset = load_dataset("MakTek/Customer_support_faqs_dataset")
    return dataset["train"]

with st.spinner("🔄 Loading FAQ dataset..."):
    dataset = load_faq_dataset()

# Extract all questions
questions = [item['question'] for item in dataset]

# Streamlit UI
st.title("🤖 Customer Support FAQ Bot")
st.write("Select a question from the list to get an answer instantly.")

# Dropdown for user to select question
selected_question = st.selectbox("📋 Choose a question:", ["-- Select a question --"] + questions)

if selected_question != "-- Select a question --":
    # Find the answer for the selected question
    answer = next((item['answer'] for item in dataset if item['question'] == selected_question), None)
    if answer:
        st.success("✅ Answer found!")
        st.markdown(f"### Answer:\n{answer}")
    else:
        st.error("❌ No answer found for the selected question.")

st.markdown("---")
st.caption("Powered by Streamlit, HuggingFace Datasets & RapidFuzz 🔥")
