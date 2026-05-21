# from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader
# from langchain.chat_models import init_chat_model
# from langchain_core.prompts import ChatPromptTemplate


# load_dotenv()

# data = TextLoader("documents._loader/notes.txt")
# docs = data.load()

# template=ChatPromptTemplate.from_messages([("system","You are the ai that suummarizes the text"),("human","{data}")])





# model = init_chat_model("mistral-small-2603")

# prompt=template.format_messages(data=docs[0].page_content)


# response = model.invoke(prompt)

# print(response.content)



# for pdf data downwards

# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chat_models import init_chat_model
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv()

# # data = PyPDFLoader("documents._loader/ML.pdf")
# # docs = data.load()


# # text_splitter = RecursiveCharacterTextSplitter(
    
# #     chunk_size=1000,
# #     chunk_overlap=200,
    
# # )
# # chunks = text_splitter.split_documents(docs)   we have used thi in cretae database 


# template=ChatPromptTemplate.from_messages([("system","You are the ai that suummarizes the text"),("human","{data}")])





# model = init_chat_model("mistral-small-2603")







from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") we can aslo use hugginface
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
llm = init_chat_model(
    model="mistral-small-latest",
    model_provider="mistralai"
)



#promt template

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant for Retrieval-Augmented Generation (RAG).

Your job is to answer the user's question using ONLY the provided context.

Instructions:
1. Read the context carefully.
2. Answer strictly based on the context.
3. If the answer is not available in the context, say:
   "I could not find the answer in the provided documents."
4. Do not make up information.
5. Keep the answer clear, accurate, and concise.
6. If possible, mention the source information from the context metadata.

Context:
{context}
"""
        ),
        (
            "human",
            """
Question:
{input}
"""
        )
    ]
)


print("Rag System Created")



print("Press 0 to exit ")



while True:
  query= input("Enter The query:-")
  if query==0:
    break
  docs=retriever.invoke(query)

  context="\n\n".join(
    [doc.page_content for doc in docs]
  )

  final_prompt=prompt.invoke({
    "context":context,
    "input":query

  })


  response = llm.invoke(final_prompt)


  print(f"\n AI:{response.content}")


