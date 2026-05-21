from langchain_huggingface import HuggingFaceEmbeddings


embedding=HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text=["hi my name is alok ranjan","how are you","i am fine bro"]


vectors = embedding.embed_documents(text)

print(vectors)