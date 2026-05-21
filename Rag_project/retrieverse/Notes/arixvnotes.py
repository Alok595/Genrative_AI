# ==============================================================
# ArxivRetriever in LangChain
# ==============================================================
#
# WHAT IS ArxivRetriever?
# --------------------------------------------------------------
# ArxivRetriever is a built-in retriever in LangChain that allows
# you to search research papers directly from arXiv.
#
# arXiv (https://arxiv.org) is a free online repository of
# scientific papers in fields such as:
# - Artificial Intelligence
# - Machine Learning
# - Mathematics
# - Physics
# - Computer Science
#
# Instead of creating your own vector database, this retriever
# fetches papers directly from arXiv based on a search query.
#
#
# USE CASES
# --------------------------------------------------------------
# - Research assistants
# - AI paper summarizers
# - Literature review tools
# - Academic search engines
# - Staying updated with latest ML papers
#
#
# WORKFLOW
# --------------------------------------------------------------
# User Query
#      ↓
# ArxivRetriever
#      ↓
# Search arXiv API
#      ↓
# Download Metadata + Abstract (or Full Paper)
#      ↓
# Return LangChain Document Objects
#
#
# WHAT DOES IT RETURN?
# --------------------------------------------------------------
# A list of Document objects.
#
# Each Document contains:
# - page_content -> abstract or full paper text
# - metadata     -> title, authors, published date, URL, etc.
#
#
# IMPORTANT PARAMETERS
# --------------------------------------------------------------
# load_max_docs
#   Number of papers to retrieve.
#
# get_full_documents
#   True  -> Download the full paper text (can be large and slower)
#   False -> Retrieve only abstracts/summary (faster)
#
#
# EXAMPLE:
# --------------------------------------------------------------
# load_max_docs=2
# → Retrieve top 2 most relevant papers.
#
# get_full_documents=True
# → Download the complete paper content.
#
# ==============================================================


# Import the built-in ArxivRetriever from LangChain Community.
from langchain_community.retrievers import ArxivRetriever


# ==============================================================
# STEP 1: CREATE THE RETRIEVER
# ==============================================================
#
# load_max_docs=2
#   Retrieve only the top 2 matching papers.
#
# get_full_documents=True
#   Download the complete content of the papers.
#   If False, only abstracts are returned.
# ==============================================================

retriever = ArxivRetriever(
    load_max_docs=2,
    get_full_documents=True,
)


# ==============================================================
# STEP 2: SEARCH FOR PAPERS
# ==============================================================
#
# Query:
# "large language model"
#
# The retriever will:
# 1. Search arXiv for papers related to large language models
# 2. Download the top 2 results
# 3. Convert them into Document objects
# ==============================================================

docs = retriever.invoke("large language model")


# ==============================================================
# STEP 3: DISPLAY RESULTS
# ==============================================================
#
# Each Document contains:
# - metadata["Title"]
# - metadata["Authors"]
# - page_content (full paper text or abstract)
#
# We print:
# - Paper title
# - Authors
# - First 500 characters of the content
# ==============================================================

for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}")
    
    # Extract metadata
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    
    # Print the beginning of the paper content
    print("Summary:", doc.page_content[:500])


# ==============================================================
# SAMPLE OUTPUT (APPROXIMATE)
# ==============================================================
#
# Result 1
# Title: Attention Is All You Need
# Authors: Ashish Vaswani, Noam Shazeer, ...
# Summary: The dominant sequence transduction models are based...
#
# Result 2
# Title: LLaMA: Open and Efficient Foundation Language Models
# Authors: Hugo Touvron, ...
# Summary: We introduce LLaMA, a collection of foundation models...
#
# ==============================================================


# ==============================================================
# COMMON METADATA FIELDS
# ==============================================================
#
# doc.metadata may contain:
# - Title
# - Authors
# - Published
# - Summary
# - Entry ID (arXiv URL)
# - Comment
# - Primary Category
#
# Example:
# print(doc.metadata)
#
# ==============================================================


# ==============================================================
# get_full_documents=True vs False
# ==============================================================
#
# True:
# - Downloads full paper text
# - Useful for deep analysis and RAG
# - Slower and larger content
#
# False:
# - Retrieves only abstracts
# - Faster and lighter
# - Good for quick summaries
#
# ==============================================================


# ==============================================================
# PRACTICAL USE CASES
# ==============================================================
#
# 1. AI Research Assistant
#    Search and summarize latest papers.
#
# 2. Paper Q&A System
#    Use papers as input to a RAG pipeline.
#
# 3. Literature Review Generator
#    Automatically summarize multiple papers.
#
# 4. Topic Explorer
#    Search for papers on "transformers", "RAG", "LLMs", etc.
#
# ==============================================================


# ==============================================================
# INTERVIEW ANSWER
# ==============================================================
#
# ArxivRetriever is a LangChain retriever that searches arXiv and
# returns scientific papers as Document objects. It can retrieve
# abstracts or full paper text, making it useful for research
# assistants, paper summarization, and academic RAG systems.
#
# ==============================================================


# ==============================================================
# MEMORY TRICK
# ==============================================================
#
# A = Academic Papers
# R = Retrieve from arXiv
# X = eXternal Research Source
#
# ==============================================================


# ==============================================================
# ONE-LINE SUMMARY
# ==============================================================
#
# ArxivRetriever allows LangChain applications to search arXiv and
# retrieve research papers as Document objects for summarization,
# analysis, or RAG workflows.
#
# ==============================================================