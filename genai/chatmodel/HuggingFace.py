
from dotenv import load_dotenv

# Loads environment variables from a .env file into the system environment
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
   
)

model = ChatHuggingFace(llm=llm)


response = model.invoke("what is langchain")

print(response.content)