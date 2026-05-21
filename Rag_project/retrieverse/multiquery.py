from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the negative gradient."
    ),
    Document(
        page_content="Gradient descent minimizes the loss function by updating model parameters based on the computed gradients."
    ),
    Document(
        page_content="Gradient descent is an optimization technique commonly used in machine learning and deep learning."
    ),
    Document(
        page_content="Neural networks use gradient descent for training by adjusting weights to reduce prediction error."
    ),
    Document(
        page_content="Support Vector Machines are supervised learning models used for classification and regression tasks."
    ),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma.from_documents(docs,embeddings)
retriver = vector_store.as_retriever()
llm=ChatMistralAI(model_name="mistral-small-latest")

multi_Query_Retriever= MultiQueryRetriever.from_llm(retriever=retriver,llm=llm)

query = "What is gradinet decent"
docs = multi_Query_Retriever.invoke(query)

print("\n Retived Documnets:\n")


for doc in docs:
  print(doc.page_content)





  