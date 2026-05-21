# streamlit_app.py

from dotenv import load_dotenv
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Initialize the model
model = init_chat_model("mistral-small-2603")

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert movie information extraction assistant.

Your task is to analyze a movie-related paragraph and extract the most useful structured information.

Present the information in a clear and well-organized format using headings and bullet points.

Include the following sections:
- Movie Name
- Director
- Main Cast
- Characters
- Genre
- Setting
- Plot Summary
- Themes
- Key Messages
- Famous Dialogues
- Character Goals
- Conflicts
- Emotional Tone
- Target Audience
- Quick Summary
- One Line Summary

Rules:
- Infer information when clearly implied.
- If information is missing, write "Not Available".
- Do not return JSON.
- Do not use markdown code blocks.
- Format the output in a clean and readable way.
        """
    ),
    (
        "human",
        """
Analyze the following movie paragraph and extract the required structured information.

Paragraph:
{paragraph}
        """
    )
])

# Streamlit UI
st.title("🎬 Movie Information Extractor")

paragraph = st.text_area(
    "Enter a movie paragraph:",
    height=300,
    placeholder="Paste a movie description here..."
)

if st.button("Extract Information"):
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph.")
    else:
        with st.spinner("Extracting movie information..."):
            final_prompt = prompt.invoke({
                "paragraph": paragraph
            })

            response = model.invoke(final_prompt)

            # Display formatted text output
            st.write(response.content)