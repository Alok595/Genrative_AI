# Load environment variables from a .env file
# This is useful for securely storing API keys and secrets
# Example: MISTRAL_API_KEY=your_key_here
from dotenv import load_dotenv
load_dotenv()


# Import required libraries

# TavilySearch can be used for web searching tools (not used in this code yet)
from langchain_tavily import TavilySearch

# ChatMistralAI is the LLM (Large Language Model) wrapper for Mistral AI
from langchain_mistralai import ChatMistralAI

# @tool decorator converts a Python function into a LangChain tool
from langchain.tools import tool

# Rich library is used for better terminal printing
from rich import print

# HumanMessage helps structure messages sent to the LLM
from langchain_core.messages import HumanMessage



# =========================================================
# 1. Creating a Custom Tool
# =========================================================

@tool
def get_text_length(text: str) -> int:
    """
    This tool returns the number of characters in a given text.
    
    Why use a tool?
    ----------------
    LLMs are good at reasoning but sometimes need external functions
    for exact operations like calculations, APIs, databases, etc.

    Here, we give the model access to a Python function
    that can count characters accurately.
    """
    
    return len(text)



# =========================================================
# Store tools in a dictionary
# =========================================================

# Why do this?
# -------------
# When the model decides to call a tool,
# it only returns the tool name as text.
#
# We use this dictionary to map the tool name
# to the actual Python function.
tools = {
    "get_text_length": get_text_length
}



# =========================================================
# Initialize the LLM
# =========================================================

# We are loading the "mistral-tiny" model
# from Mistral AI using LangChain.
llm = ChatMistralAI(model="mistral-tiny")



# =========================================================
# Bind tools to the model
# =========================================================

# Why bind tools?
# ----------------
# This tells the LLM:
# "You are allowed to use these tools if needed."
#
# Without binding, the model cannot call tools.
llm_with_tool = llm.bind_tools([get_text_length])



# =========================================================
# Create a message history list
# =========================================================

# This list stores the conversation history.
# LLMs work better when previous messages are preserved.
message = []



# =========================================================
# Take user input
# =========================================================

question = input("You:-")



# =========================================================
# Convert user input into HumanMessage
# =========================================================

# LangChain expects messages in structured format.
query = HumanMessage(question)

# Add user message to conversation history
message.append(query)



# =========================================================
# First LLM Call
# =========================================================

# The model checks:
# - Does it need a tool?
# - Or can it answer directly?
result = llm_with_tool.invoke(message)

# Save model response
message.append(result)



# =========================================================
# Check if the model requested a tool
# =========================================================

if result.tool_calls:

    # Extract the tool name requested by the model
    tool_name = result.tool_calls[0]["name"]

    
    # Execute the tool
    # ----------------
    # The model provides arguments automatically.
    #
    # Example:
    # {
    #   "name": "get_text_length",
    #   "args": {"text": "hello"}
    # }
    #
    # invoke() runs the actual Python function.
    tool_message = tools[tool_name].invoke(result.tool_calls[0])

    
    # Add tool result to message history
    # This allows the LLM to see the tool output
    message.append(tool_message)



# =========================================================
# Second LLM Call
# =========================================================

# Now the model has:
# - User question
# - Tool call
# - Tool output
#
# It can generate the final answer.
result = llm_with_tool.invoke(message)



# =========================================================
# Print Final Output
# =========================================================

print(result.content)