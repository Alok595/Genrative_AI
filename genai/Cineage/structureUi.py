# streamlit_app.py

from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables from .env
load_dotenv()

# Initialize the model
model = init_chat_model("mistral-small-2603")


# Define the Pydantic schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# Create the output parser
parser = PydanticOutputParser(pydantic_object=Movie)


# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the paragraph.

{format_instructions}
        """
    ),
    (
        "human",
        "{paragraph}"
    )
])


# Streamlit UI
st.title("🎬 Structured Movie Information Extractor")

paragraph = st.text_area(
    "Enter a movie paragraph:",
    height=250,
    placeholder="Paste a movie description here..."
)

if st.button("Extract Information"):
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph.")
    else:
        with st.spinner("Extracting structured movie information..."):
            # Create final prompt
            final_prompt = prompt.invoke({
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions()
            })

            # Invoke model
            response = model.invoke(final_prompt)

            # Parse into Pydantic object
            movie = parser.parse(response.content)

            # Display as formatted JSON
            st.json(movie.model_dump())