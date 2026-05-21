# ==============================================================
# MultiQueryRetriever Example with Detailed Notes
# ==============================================================
#
# WHAT IS MultiQueryRetriever?
# --------------------------------------------------------------
# MultiQueryRetriever is an advanced retriever in LangChain.
#
# Problem:
# A user may ask a question with spelling mistakes, synonyms,
# or different wording than what exists in the documents.
#
# Example:
# User Query: "What is gradinet decent"   (misspelled)
#
# If we use normal retrieval:
# - The retriever searches only using this exact query embedding.
# - It may miss some relevant documents.
#
# If we use MultiQueryRetriever:
# - An LLM rewrites the query into multiple better versions:
#     1. What is gradient descent?
#     2. Explain gradient descent in machine learning.
#     3. How does gradient descent minimize loss?
# - Each query searches the vector database.
# - Results are merged and duplicates removed.
#
# BENEFITS:
# - Handles spelling mistakes
# - Handles synonyms
# - Improves recall (finds more relevant documents)
# - Better answers in RAG systems
#
# WORKFLOW:
# --------------------------------------------------------------
# User Query
#     ↓
# LLM Generates Multiple Query Variations
#     ↓
# Search Vector Store for Each Query
#     ↓
# Merge and Remove Duplicate Documents
#     ↓
# Return Final Relevant Documents
#
# ==============================================================


# Import LangChain's standard Document object.
# Each Document contains:
# - page_content -> actual text
# - metadata     -> optional extra information
from langchain_core.documents import Document


# Chroma is a local vector database used to store embeddings
# and perform similarity search.
from langchain_community.vectorstores import Chroma


# HuggingFaceEmbeddings converts text into vectors (embeddings).
# We are using a free local embedding model from Hugging Face.
from langchain_community.embeddings import HuggingFaceEmbeddings


# MultiQueryRetriever automatically generates multiple versions
# of the user's query using an LLM.
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# ChatMistralAI is the chat model used to generate alternative
# search queries.
from langchain_mistralai import ChatMistralAI


# load_dotenv() loads API keys from the .env file.
from dotenv import load_dotenv


# Load environment variables such as MISTRAL_API_KEY
load_dotenv()


# ==============================================================
# STEP 1: CREATE DOCUMENTS
# ==============================================================
#
# Normally these documents come from:
# - PDFs
# - Websites
# - Text files
# - Databases
#
# For this demo, we manually create a few documents.
# Most of them are about Gradient Descent.
# One document is about Support Vector Machines (irrelevant).
#
# The retriever should return only the relevant documents.
# ==============================================================

docs = [
    Document(
        page_content="Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the negative gradient."
    ),
    Document(
        page_content="Gradient descent minimizes the loss function by updating model parameters based on the computed gradients."
    ),
    Document(
        page_content="Gradient descent is an optimization technique commonly used in machine learning and deep learning."
    ),
    Document(
        page_content="Neural networks use gradient descent for training by adjusting weights to reduce prediction error."
    ),
    Document(
        page_content="Support Vector Machines are supervised learning models used for classification and regression tasks."
    ),
]


# ==============================================================
# STEP 2: INITIALIZE EMBEDDING MODEL
# ==============================================================
#
# Embeddings convert text into numerical vectors.
#
# Similar meanings → similar vectors.
#
# Example:
# "What is gradient descent?"
# and
# "Explain gradient descent"
# will have similar embeddings.
#
# Model Used:
# sentence-transformers/all-mpnet-base-v2
#
# This is one of the best free embedding models available.
# ==============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ==============================================================
# STEP 3: CREATE VECTOR STORE
# ==============================================================
#
# Chroma stores:
# - Document text
# - Document embeddings
# - Metadata (if any)
#
# Chroma.from_documents():
# 1. Converts each document into embeddings
# 2. Stores them in Chroma DB
#
# ==============================================================

vector_store = Chroma.from_documents(
    docs,
    embeddings
)


# ==============================================================
# STEP 4: CREATE BASIC RETRIEVER
# ==============================================================
#
# A retriever is a search interface over the vector database.
#
# retriever.invoke(query)
# → returns the most relevant documents.
#
# This is a standard similarity search retriever.
# ==============================================================

retriever = vector_store.as_retriever()


# ==============================================================
# STEP 5: INITIALIZE LLM
# ==============================================================
#
# MultiQueryRetriever requires an LLM to generate alternative
# versions of the user's query.
#
# Example:
# "What is gradinet decent"
# becomes:
# - What is gradient descent?
# - Explain gradient descent.
# - How does gradient descent work?
#
# Model Used:
# mistral-small-latest
# ==============================================================

llm = ChatMistralAI(
    model_name="mistral-small-latest"
)


# ==============================================================
# STEP 6: CREATE MultiQueryRetriever
# ==============================================================
#
# from_llm() combines:
# - Base retriever
# - LLM for query rewriting
#
# The resulting retriever:
# 1. Receives the user query
# 2. Generates multiple reformulated queries
# 3. Searches the vector store for each query
# 4. Merges unique documents
# 5. Returns the final results
# ==============================================================

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# ==============================================================
# STEP 7: USER QUERY
# ==============================================================
#
# Intentional spelling mistake:
# "gradinet decent"
#
# MultiQueryRetriever should still understand that the user
# means "gradient descent".
# ==============================================================

query = "What is gradinet decent"


# ==============================================================
# STEP 8: RETRIEVE DOCUMENTS
# ==============================================================
#
# invoke(query) triggers the full workflow:
#
# 1. LLM rewrites the query
# 2. Multiple searches are performed
# 3. Duplicate documents are removed
# 4. Relevant documents are returned
# ==============================================================

retrieved_docs = multi_query_retriever.invoke(query)


# ==============================================================
# STEP 9: DISPLAY RESULTS
# ==============================================================
#
# We print all retrieved documents.
# The SVM document should usually not appear because it is
# unrelated to the query.
# ==============================================================

print("\nRetrieved Documents:\n")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 80)


# ==============================================================
# EXPECTED OUTPUT (APPROXIMATE)
# ==============================================================
#
# Document 1:
# Gradient descent is an optimization algorithm used to minimize...
#
# Document 2:
# Gradient descent minimizes the loss function...
#
# Document 3:
# Gradient descent is an optimization technique...
#
# Document 4:
# Neural networks use gradient descent for training...
#
# ==============================================================
# INTERVIEW ANSWER
# ==============================================================
#
# MultiQueryRetriever is a LangChain retriever that uses an LLM
# to generate multiple alternative versions of the user's query.
# It performs retrieval for each version, merges the results,
# and returns the most relevant unique documents, improving
# recall and handling synonyms, paraphrases, and spelling errors.
#
# ==============================================================
# MEMORY TRICK
# ==============================================================
#
# Q → Query
# R → Rewrite into multiple forms
# S → Search multiple times
# M → Merge results
#
# "Rewrite, Search, Merge"
#
# ==============================================================