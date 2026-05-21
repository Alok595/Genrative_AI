# ==============================================================
# Similarity Search vs MMR Retriever in LangChain
# ==============================================================
#
# In LangChain, a Vector Store (like Chroma) can be converted into
# different types of retrievers.
#
# Two of the most common retrievers are:
#
# 1. Similarity Search Retriever
# 2. MMR Retriever (Maximum Marginal Relevance)
#
# --------------------------------------------------------------
# 1. Similarity Search
# --------------------------------------------------------------
# Returns the top-k documents that are most similar to the query.
#
# Characteristics:
# - Focuses only on similarity score.
# - May return duplicate or highly similar documents.
# - Fast and commonly used.
#
# Example:
# Query: "What is gradient descent?"
#
# Results:
# - Gradient descent is an optimization algorithm...
# - Gradient descent minimizes the loss function...
# - Gradient descent is an optimization technique...
#
# These results are all highly similar.
#
#
# --------------------------------------------------------------
# 2. MMR (Maximum Marginal Relevance)
# --------------------------------------------------------------
# Returns documents that are:
# - Relevant to the query
# - Diverse from each other
#
# Characteristics:
# - Reduces redundancy
# - Provides broader coverage
# - Useful when documents contain repeated information
#
# Example:
# Query: "What is gradient descent?"
#
# Results:
# - Gradient descent is an optimization algorithm...
# - Neural networks use gradient descent for training...
# - Gradient descent is commonly used in machine learning...
#
# Notice the documents cover different aspects.
#
#
# --------------------------------------------------------------
# Similarity vs MMR
# --------------------------------------------------------------
#
# Similarity Search:
#   "Give me the most similar documents."
#
# MMR:
#   "Give me relevant documents, but avoid repeating the same idea."
#
#
# --------------------------------------------------------------
# WHEN TO USE WHICH?
# --------------------------------------------------------------
#
# Use Similarity Search when:
# - You want the most relevant chunks
# - Speed is important
# - Redundancy is acceptable
#
# Use MMR when:
# - Documents contain repeated information
# - You want diverse context
# - Building RAG systems with broader coverage
#
#
# --------------------------------------------------------------
# WORKFLOW
# --------------------------------------------------------------
#
# Query
#   ↓
# Convert Query to Embedding
#   ↓
# Compare with Stored Embeddings
#   ↓
# Search Strategy
#   ├── Similarity → Top most similar chunks
#   └── MMR        → Relevant + Diverse chunks
#   ↓
# Return Documents
#
# ==============================================================


# Import the standard Document object used in LangChain.
# Each Document contains:
# - page_content : actual text
# - metadata     : optional additional information
from langchain_core.documents import Document


# Chroma is a local vector database used to:
# - Store embeddings
# - Perform similarity search
from langchain_community.vectorstores import Chroma


# HuggingFaceEmbeddings converts text into vector embeddings.
# We are using a free local embedding model.
from langchain_huggingface import HuggingFaceEmbeddings


# ==============================================================
# STEP 1: CREATE SAMPLE DOCUMENTS
# ==============================================================
#
# Four documents discuss Gradient Descent.
# One document discusses Support Vector Machines.
#
# The retriever should return only documents relevant to the query.
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
# STEP 2: LOAD EMBEDDING MODEL
# ==============================================================
#
# This model converts text into dense vectors.
#
# Similar meaning → similar vectors.
#
# Model:
# sentence-transformers/all-mpnet-base-v2
#
# A high-quality free embedding model from Hugging Face.
# ==============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ==============================================================
# STEP 3: CREATE VECTOR STORE
# ==============================================================
#
# Chroma.from_documents():
# 1. Generates embeddings for all documents
# 2. Stores them in Chroma DB
# 3. Enables semantic search
# ==============================================================

vector_store = Chroma.from_documents(
    docs,
    embeddings
)


# ==============================================================
# STEP 4: CREATE SIMILARITY RETRIEVER
# ==============================================================
#
# search_type="similarity"
# → Returns the top-k most similar documents.
#
# search_kwargs={"k": 3}
# → Return top 3 documents.
# ==============================================================

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


print("========== SIMILARITY SEARCH RESULTS ==========\n")


# Retrieve documents using similarity search.
# Note: "decent" is a misspelling of "descent".
# Embeddings are robust enough to still find relevant documents.
similarity_docs = similarity_retriever.invoke(
    "What is gradient decent"
)


# Print retrieved documents
for i, doc in enumerate(similarity_docs, start=1):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 80)


# ==============================================================
# STEP 5: CREATE MMR RETRIEVER
# ==============================================================
#
# search_type="mmr"
# → Maximum Marginal Relevance.
#
# Goal:
# - High relevance to query
# - Low redundancy between results
#
# search_kwargs={"k": 3}
# → Return 3 diverse relevant documents.
# ==============================================================

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)


print("\n========== MMR RESULTS ==========\n")


# Retrieve documents using MMR
mmr_docs = mmr_retriever.invoke(
    "What is gradient decent?"
)


# Print retrieved documents
for i, doc in enumerate(mmr_docs, start=1):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 80)


# ==============================================================
# EXPECTED DIFFERENCE
# ==============================================================
#
# Similarity Search:
# - May return three documents that all define gradient descent.
#
# MMR:
# - May return:
#   1. Definition of gradient descent
#   2. Use in neural networks
#   3. Use in machine learning
#
# This gives more diverse and informative context.
#
# ==============================================================
# INTERVIEW ANSWER
# ==============================================================
#
# Similarity retriever returns the top-k most semantically similar
# documents to the query. MMR retriever returns documents that are
# both relevant and diverse, reducing redundancy and improving the
# quality of retrieved context for RAG applications.
#
# ==============================================================
# MEMORY TRICK
# ==============================================================
#
# Similarity  → "Most Similar"
# MMR         → "Relevant + Diverse"
#
# ==============================================================