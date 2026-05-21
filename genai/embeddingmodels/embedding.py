from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Loads environment variables from a .env file into the system environment
load_dotenv()

embeddings = OpenAIEmbeddings(models="text-embedding-3-large",dimensions=64)

# text=["hi my name is alok ranjan","how are you","i am fine bro"]

# vectors = embeddings.embed_documents(text)

vectors = embeddings.embed_query("Hi my name is alok")

print(vectors)


#open ai is paid

