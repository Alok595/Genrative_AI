import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -----------------------------
# Load LLM
# -----------------------------
llm = init_chat_model(
    model="mistral-small-latest",
    model_provider="mistralai"
)

# -----------------------------
# Embeddings + Vector DB
# -----------------------------
embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a RAG assistant.
Answer ONLY from the provided context.
If not found, say: "I could not find the answer in the provided documents."

Context:
{context}
"""),
    ("human", "{input}")
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")

st.title("📄 RAG PDF Chat App (Mistral + Chroma)")

# -----------------------------
# PDF Upload Section
# -----------------------------
st.sidebar.header("📤 Upload PDF")

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Store in Chroma
    vector_store.add_documents(chunks)

    st.sidebar.success("PDF processed & stored in vector DB!")

    os.remove(file_path)

# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question")

if st.button("Ask") and query:

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "input": query
    })

    response = llm.invoke(final_prompt)

    st.markdown("### 🤖 Answer")
    st.write(response.content)

# -----------------------------
# Debug (optional)
# -----------------------------
with st.expander("Show retrieved chunks"):
    if query:
        docs = retriever.invoke(query)
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(d.page_content)