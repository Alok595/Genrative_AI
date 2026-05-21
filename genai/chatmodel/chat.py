from dotenv import load_dotenv

# Loads environment variables from a .env file into the system environment
load_dotenv()


"""
------------------------------------------------------------
LangChain Chat Models - init_chat_model Notes
------------------------------------------------------------

1. OPENAI (via LangChain)
--------------------------------
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1")

- Uses OpenAI models
- Requires OPENAI_API_KEY in .env file
- Example models: gpt-4.1, gpt-4o, etc.


2. GOOGLE GEMINI
--------------------------------
from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

- Uses Google Gemini models
- Requires GOOGLE_API_KEY in .env file
- Fast and lightweight model: flash-lite variants


3. GROQ (FAST INFERENCE)
--------------------------------
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:openai/gpt-oss-120b")

- Uses Groq inference engine
- Very fast response time
- Requires GROQ_API_KEY in .env file
- Good for low-latency LLM apps

4. Mistral
from langchain.chat_models import init_chat_model

model = init_chat_model("mistral-small-2603")

------------------------------------------------------------
HOW TO USE MODEL
------------------------------------------------------------

response = model.invoke("Hello! What is Generative AI?")
print(response.content)

- .invoke() sends a prompt to the model
- response.content contains the generated answer


------------------------------------------------------------
NOTES
------------------------------------------------------------

- init_chat_model() automatically selects provider based on prefix:
    "gpt-..." -> OpenAI
    "google_genai:..." -> Gemini
    "groq:..." -> Groq

- Always keep API keys in .env file (never hardcode)
- load_dotenv() loads them automatically
"""


from langchain.chat_models import init_chat_model

model = init_chat_model("mistral-small-2603")

response = model.invoke("Hello! What is Generative AI?")
print(response.content)