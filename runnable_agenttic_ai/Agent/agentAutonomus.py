# =========================================================
# AGENTIC AI PROJECT
# CITY ASSISTANT WITH TOOL CALLING + HUMAN APPROVAL
# =========================================================

# This project demonstrates:
#
# ✅ Tool Calling
# ✅ API Integration
# ✅ Human-in-the-loop AI
# ✅ LangChain Agents
# ✅ Middleware
# ✅ Weather API
# ✅ News Search
# ✅ AI Decision Making
#
# FLOW:
#
# User
#   ↓
# Agent (LLM)
#   ↓
# Decide if tool is needed
#   ↓
# Human Approval Middleware
#   ↓
# Tool Execution
#   ↓
# Tool Result
#   ↓
# LLM Generates Final Answer
#
# =========================================================



# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

# load_dotenv() reads variables from .env file
#
# Example .env:
#
# OPENWEATHER_API_KEY=xxxxx
# TAVILY_API_KEY=xxxxx
#
# Without this:
# os.getenv() will return None

from dotenv import load_dotenv

load_dotenv()



# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
# Used to access environment variables

import requests
# Used for making HTTP API requests

from tavily import TavilyClient
# Tavily search client for internet/news search

from langchain_mistralai import ChatMistralAI
# Mistral LLM integration

from langchain.tools import tool
# Converts python functions into AI callable tools

from langchain_core.messages import ToolMessage
# Used for sending tool results back to the agent

from rich import print
# Better terminal output formatting

from langchain.agents import create_agent
# Creates an AI agent with tool calling abilities

from langchain.agents.middleware import wrap_tool_call
# Middleware system for intercepting tool calls



# =========================================================
# WEATHER TOOL
# =========================================================

# @tool converts this function into an AI tool
#
# Now the LLM can automatically decide:
#
# "I should call get_weather()"

@tool
def get_weather(city: str) -> str:

    """
    Get current weather of a city
    """

    # Read OpenWeather API key from .env
    API_KEY = os.getenv("OPENWEATHER_API_KEY")



    # =====================================================
    # OPENWEATHER API URL
    # =====================================================

    # q = city name
    # appid = API key
    # units=metric gives temperature in Celsius

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )



    # =====================================================
    # SEND API REQUEST
    # =====================================================

    # requests.get() sends HTTP GET request
    response = requests.get(url)



    # =====================================================
    # CONVERT RESPONSE TO JSON
    # =====================================================

    # API response comes as JSON
    # json() converts it into Python dictionary

    data = response.json()



    # =====================================================
    # DEBUGGING
    # =====================================================

    # Prints complete API response

    print("DEBUG:", data)



    # =====================================================
    # ERROR HANDLING
    # =====================================================

    # OpenWeather returns:
    #
    # cod = 200 → success
    # other value → error

    if str(data.get("cod")) != "200":

        return (
            f"Error: "
            f"{data.get('message', 'Could not fetch weather')}"
        )



    # =====================================================
    # EXTRACT IMPORTANT DATA
    # =====================================================

    # Temperature

    temp = data["main"]["temp"]

    # Weather description

    desc = data["weather"][0]["description"]



    # =====================================================
    # RETURN CLEAN RESPONSE
    # =====================================================

    return f"Weather in {city}: {desc}, {temp}°C"



# =========================================================
# TAVILY CLIENT SETUP
# =========================================================

# Tavily is used for real-time web search/news

client = TavilyClient(

    # Read Tavily API key from .env
    api_key=os.getenv("TAVILY_API_KEY")
)



# =========================================================
# NEWS TOOL
# =========================================================

# Another AI callable tool

@tool
def get_news(city: str) -> str:

    """
    Get latest news about the city
    """



    # =====================================================
    # CREATE SEARCH QUERY
    # =====================================================

    query = f"Latest news about {city}"



    # =====================================================
    # SEARCH INTERNET USING TAVILY
    # =====================================================

    response = client.search(

        # Search query
        query=query,

        # basic = faster
        # advanced = deeper search
        search_depth="basic",

        # Number of results
        max_results=3
    )



    # =====================================================
    # EXTRACT SEARCH RESULTS
    # =====================================================

    results = response.get("results", [])



    # =====================================================
    # HANDLE EMPTY RESULTS
    # =====================================================

    if not results:
        return f"No news found for {city}"



    # =====================================================
    # FORMAT NEWS
    # =====================================================

    news_list = []



    # Loop through every article
    for item in results:

        # Extract title
        title = item.get("title", "")

        # Extract article content/summary
        content = item.get("content", "")

        # Extract article URL
        url = item.get("url", "")



        # ================================================
        # CREATE CLEAN ARTICLE FORMAT
        # ================================================

        news_list.append(

            f"Title: {title}\n"
            f"Summary: {content}\n"
            f"Source: {url}\n"
        )



    # =====================================================
    # RETURN ALL NEWS ARTICLES
    # =====================================================

    return "\n\n".join(news_list)



# =========================================================
# TEST TOOL DIRECTLY
# =========================================================

# invoke() manually runs tool
#
# Useful for debugging

print(get_news.invoke("Ranchi"))



# =========================================================
# CREATE LLM
# =========================================================

# Mistral AI model

llm = ChatMistralAI(

    # Small fast model
    model="mistral-tiny"
)



# =========================================================
# HUMAN APPROVAL MIDDLEWARE
# =========================================================

# Middleware acts like:
#
# "interceptor"
#
# It runs BEFORE tool execution.
#
# Here:
# We ask user permission before every tool call.

@wrap_tool_call
def humman_approval(request, handler):

    """
    Ask human approval before every tool call
    """



    # =====================================================
    # GET TOOL NAME
    # =====================================================

    tool_name = request.tool_call["name"]



    # =====================================================
    # ASK USER PERMISSION
    # =====================================================

    confirm = input(

        f"Agent wants to call '{tool_name}'. "
        f"Approve (yes/no): "
    )



    # =====================================================
    # IF USER DENIES TOOL
    # =====================================================

    if confirm.lower() != "yes":

        # ToolMessage tells agent:
        # tool execution was denied

        return ToolMessage(

            content="Tool Call Denied by the user",

            tool_call_id=request.tool_call['id']
        )



    # =====================================================
    # EXECUTE TOOL
    # =====================================================

    # handler(request)
    #
    # passes control back to LangChain
    # and actually executes the tool

    return handler(request)



# =========================================================
# CREATE AGENT
# =========================================================

# create_agent() automatically creates:
#
# ✅ Tool calling loop
# ✅ ReAct reasoning
# ✅ Tool execution
# ✅ Tool result handling
# ✅ LLM orchestration

agent = create_agent(

    # LLM brain
    llm,

    # Available tools
    tools=[get_news, get_weather],

    # Permanent AI instructions
    system_prompt="You are a helpful city assistant",

    # Middleware list
    middleware=[
        humman_approval
    ],
)



# =========================================================
# TERMINAL UI
# =========================================================

print("City Agent | type exit to exit")



# =========================================================
# MAIN CHAT LOOP
# =========================================================

while True:



    # =====================================================
    # GET USER INPUT
    # =====================================================

    user_input = input("You: ")



    # =====================================================
    # EXIT CONDITION
    # =====================================================

    if user_input.lower() == "exit":
        break



    # =====================================================
    # SEND MESSAGE TO AGENT
    # =====================================================

    # messages format follows chat API structure

    result = agent.invoke(

        {
            "messages": [

                {
                    "role": "user",

                    "content": user_input
                }
            ]
        }
    )



    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    # result contains many messages internally
    #
    # Last message usually contains:
    # final AI response

    print(result['messages'][-1].content)



# =========================================================
# COMPLETE INTERNAL FLOW
# =========================================================

# User:
# "What's weather in Ranchi?"
#
# Step 1:
# User message sent to LLM
#
# Step 2:
# LLM thinks:
# "Need weather data"
#
# Step 3:
# LLM decides:
# call get_weather()
#
# Step 4:
# Middleware intercepts tool call
#
# Step 5:
# Human approval requested
#
# Step 6:
# If approved:
# tool executes
#
# Step 7:
# OpenWeather API returns data
#
# Step 8:
# Tool result sent back to LLM
#
# Step 9:
# LLM generates natural language response
#
# Final Output:
# "Current weather in Ranchi is cloudy with 28°C."
#
# =========================================================