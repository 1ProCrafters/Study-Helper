import streamlit as st
from transformers import pipeline
import pandas as pd

# Load Hugging Face pipelines (no API key required)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")
text_to_speech = pipeline("text-to-speech", model="microsoft/DialoGPT-small")

# Initialize or retrieve flashcards and performance data
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "performance" not in st.session_state:
    st.session_state.performance = pd.DataFrame(columns=["Session", "Score"])
if "notes" not in st.session_state:
    st.session_state.notes = ""

# App Title
st.title("AI-Powered Study Assistant")

# Sidebar for tool selection
st.sidebar.header("Choose Your Study Tool")
tool = st.sidebar.selectbox(
    "Select a tool", 
    ["Summarize Text", "Generate Practice Questions", "Create Flashcards"]
)

# Summarize Text
if tool == "Summarize Text":
    st.header("Summarize Your Study Materials")
    text_input = st.text_area("Enter text to be summarized:")
    if st.button("Summarize"):
        summary = summarizer(text_input, max_length=100, min_length=25, do_sample=False)[0]["summary_text"]
        st.write("**Summary:**", summary)

# Generate Practice Questions
elif tool == "Generate Practice Questions":
    st.header("Generate Practice Questions from Text")
    text_input = st.text_area("Enter text to generate questions from:")
    if st.button("Generate Questions"):
        questions = question_generator("generate question: " + text_input)
        for i, q in enumerate(questions):
            st.write(f"**Question {i + 1}:** {q['generated_text']}")

# Create Flashcards and Review with Spaced Repetition
elif tool == "Create Flashcards":
    st.header("Create Flashcards for Active Recall")
    st.write("Enter each flashcard in the format: `Front:Back`")
    flashcard_input = st.text_area("Add your flashcards here:")
    if st.button("Save Flashcards"):
        new_flashcards = [line.split(":") for line in flashcard_input.split("\n") if ":" in line]
        st.session_state.flashcards.extend(new_flashcards)
        st.write("Flashcards saved!")

    # Review flashcards
    st.subheader("Flashcard Review Session")
    if st.session_state.flashcards:
        for idx, (front, back) in enumerate(st.session_state.flashcards):
            with st.expander(f"Flashcard {idx + 1}"):
                st.write(f"**Front:** {front}")
                if st.button(f"Show Answer {idx}", key=f"show_{idx}"):
                    st.write(f"**Back:** {back}")


