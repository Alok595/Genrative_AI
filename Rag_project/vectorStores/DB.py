from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document


docs = [
    Document(
        page_content="Fake news spreads quickly on social media platforms.",
        metadata={"source": "ML"}
    ),
    Document(
        page_content="SMS scams often contain urgent language and suspicious links.",
        metadata={"source": "DL"}
    ),
    Document(
        page_content="Machine learning models can classify text as fake or real.",
        metadata={"source": "NLP"}
    ),
]


embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

# --------------------------------------------------
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


result=vector_store.similarity_search("which model classify fake and real",k=2)

for i in result:
  print(i.page_content)
  print(i.metadata)


retrivers = vector_store.as_retriever()


docs=retrivers.invoke("Explain Machine learning")

for d in docs:
  print(d.page_content)

