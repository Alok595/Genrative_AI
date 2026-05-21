# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import required libraries
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print


#1 Creating a tool

@tool
def get_text_length(text:str)->int:
  """Genrate the number of charter in a given text"""
  return len(text)

llm=ChatMistralAI(model="mistral-tiny")


#tool binding
llm_with_tool=llm.bind_tools([get_text_length])

result=llm.invoke("hellow")

result2= llm_with_tool.invoke("hellow")

print(result)

print()
print()

print()
print()

print(result2)