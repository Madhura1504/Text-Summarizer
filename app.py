# app.py

import streamlit as st
from main import extractive_summary, abstractive_summary

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("📝 Text Summarization Tool")
st.write("Summarize long text using **Extractive (TextRank)** or **Abstractive (T5)** methods.")

text_input = st.text_area("Enter your text here:", height=300)

method = st.radio("Choose summarization method:", ["Extractive (TextRank)", "Abstractive (T5)"])

if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing..."):
            if method == "Extractive (TextRank)":
                summary = extractive_summary(text_input, num_sentences=3)
            else:
                summary = abstractive_summary(text_input)
        st.success("Summary:")
        st.write(summary)
