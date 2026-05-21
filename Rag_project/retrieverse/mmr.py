from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


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

similarity_retriever = vector_store.as_retriever(

  search_type="similarity",
  search_kwargs={'k':3}
)


print("Similarity Search Results ")

similarity_docs = similarity_retriever.invoke("What is gradient decent")


for doc in similarity_docs:
  print(doc.page_content)


mmr_retriver = vector_store.as_retriever(
   search_type="mmr",
   search_kwargs={'k':3}
)


print(" MMR RESULTS")


mmr_docs = mmr_retriver.invoke("What is gradient decent ?")
for doc in mmr_docs:
  print(doc.page_content)