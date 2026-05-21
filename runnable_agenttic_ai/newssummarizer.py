"""
AI News Summarizer using LangChain, Tavily Search, and Mistral AI
---------------------------------------------------------------

Requirements:
    pip install python-dotenv langchain langchain-core langchain-tavily langchain-mistralai

Environment Variables (.env):
    TAVILY_API_KEY=your_tavily_api_key
    MISTRAL_API_KEY=your_mistral_api_key
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import required libraries
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# Step 1: Initialize Tavily Search Tool
# ---------------------------------------------------------
# max_results specifies how many search results to retrieve
search_tool = TavilySearch(max_results=5)


# ---------------------------------------------------------
# Step 2: Initialize Mistral Language Model
# ---------------------------------------------------------
# Using the lightweight "mistral-tiny" model
llm = ChatMistralAI(model="mistral-tiny")


# ---------------------------------------------------------
# Step 3: Create Prompt Template
# ---------------------------------------------------------
# This prompt instructs the LLM to summarize the news
# into concise and easy-to-read bullet points.
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Summarize the following AI news into clear and concise bullet points.

News:
{news}
""")


# ---------------------------------------------------------
# Step 4: Create Output Parser
# ---------------------------------------------------------
# Converts the model response into plain text
output_parser = StrOutputParser()


# ---------------------------------------------------------
# Step 5: Build LangChain Pipeline
# ---------------------------------------------------------
# Flow:
# Prompt Template -> LLM -> Output Parser
chain = prompt | llm | output_parser


# ---------------------------------------------------------
# Step 6: Search for Latest AI News
# ---------------------------------------------------------
news_result = search_tool.run("Latest AI news of 2026")


# ---------------------------------------------------------
# Step 7: Generate Summary
# ---------------------------------------------------------
result = chain.invoke({"news": news_result})


# ---------------------------------------------------------
# Step 8: Print Final Output
# ---------------------------------------------------------
print(result)



