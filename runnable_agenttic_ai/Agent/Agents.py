# ================================
# LOAD ENVIRONMENT VARIABLES
# ================================

# load_dotenv() reads the .env file
# and loads API keys into environment variables

from dotenv import load_dotenv
load_dotenv()


# ================================
# IMPORT LIBRARIES
# ================================

import os                     # access environment variables
import requests               # make HTTP API requests

# Tavily search client
from tavily import TavilyClient

# Mistral LLM model
from langchain_mistralai import ChatMistralAI

# LangChain tool decorator
from langchain.tools import tool

# Message types used in conversation memory
from langchain_core.messages import HumanMessage, ToolMessage

# Rich print for better terminal formatting
from rich import print



# ================================
# WEATHER TOOL
# ================================

# @tool converts a normal python function
# into an AI callable tool

@tool
def get_weather(city: str) -> str:

    """
    Get current weather of a city
    """

    # Read OpenWeather API key from .env
    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    # OpenWeather API URL
    # units=metric gives temperature in Celsius
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    # Send GET request to API
    response = requests.get(url)

    # Convert response into JSON dictionary
    data = response.json()

    # Debugging output
    print("DEBUG:", data)

    # Check if API returned success code
    # OpenWeather returns:
    # cod = 200 → success
    # otherwise error

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    # Extract temperature
    temp = data["main"]["temp"]

    # Extract weather description
    desc = data["weather"][0]["description"]

    # Return clean formatted response
    return f"Weather in {city}: {desc}, {temp}°C"



# ================================
# TAVILY CLIENT SETUP
# ================================

# Create Tavily search client
# API key comes from .env

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)



# ================================
# NEWS TOOL
# ================================

@tool
def get_news(city: str) -> str:

    """
    Get latest news about the city
    """

    # Create search query
    query = f"Latest news about {city}"

    # Search internet using Tavily
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3
    )

    # Extract results list
    results = response.get("results", [])

    # If no news found
    if not results:
        return f"No news found for {city}"

    # Store formatted news
    news_list = []

    # Loop through each news article
    for item in results:

        # Extract title
        title = item.get("title", "")

        # Extract article summary/content
        content = item.get("content", "")

        # Extract source URL
        url = item.get("url", "")

        # Format article nicely
        news_list.append(
            f"Title: {title}\n"
            f"Summary: {content}\n"
            f"Source: {url}\n"
        )

    # Combine all news articles into one string
    return "\n\n".join(news_list)



# ================================
# TEST TOOL DIRECTLY
# ================================

# invoke() manually executes the tool

print(get_news.invoke("Ranchi"))



# ================================
# CREATE LLM
# ================================

# Initialize Mistral model

llm = ChatMistralAI(
    model="mistral-tiny"
)



# ================================
# TOOL DICTIONARY
# ================================

# Dictionary used for executing tools dynamically

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}



# ================================
# BIND TOOLS TO LLM
# ================================

# This gives the LLM access to tools
# Now the model can decide:
# "Should I call a tool?"

llm_with_tool = llm.bind_tools(
    [get_weather, get_news]
)



# ================================
# AGENT MEMORY
# ================================

# messages stores conversation history
# This acts as memory/context

messages = []



# ================================
# TERMINAL UI
# ================================

print("City Intelligent System")
print("Press Exit to quit")



# ================================
# MAIN AGENT LOOP
# ================================

while True:

    # Get user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == "exit":
        break

    # Add user message into memory
    messages.append(
        HumanMessage(content=user_input)
    )



    # ================================
    # AGENT THINKING LOOP
    # ================================

    while True:

        # Send conversation history to LLM
        result = llm_with_tool.invoke(messages)

        # Store AI response into memory
        messages.append(result)



        # ================================
        # CHECK TOOL CALLS
        # ================================

        # If AI wants to use a tool
        if result.tool_calls:

            # Loop through requested tools
            for tool_call in result.tool_calls:

                # Get tool name
                tool_name = tool_call["name"]



                # ================================
                # HUMAN IN THE LOOP
                # ================================

                # Ask human permission before tool execution

                confirm = input(
                    f"Agent wants to call {tool_name}. Approve (y/n): "
                )



                # IMPORTANT:
                # confirm.lower is WRONG
                # must use confirm.lower()

                if confirm.lower() == "n":

                    print(
                        "Tool call denied. "
                        "Cannot fetch latest information."
                    )

                    break



                # ================================
                # EXECUTE TOOL
                # ================================

                # Run selected tool dynamically

                tool_result = tools[tool_name].invoke(
                    tool_call
                )



                # ================================
                # SEND TOOL RESULT BACK TO LLM
                # ================================

                # ToolMessage lets the LLM read
                # the tool output

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                )

            # Continue agent loop
            # so LLM can process tool result
            continue



        # ================================
        # FINAL RESPONSE
        # ================================

        else:

            # No more tool calls
            # Print final AI response

            print(result.content)

            break