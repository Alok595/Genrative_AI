#load pdf
#split in docs
#create the embeding
#store into chroma




from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()


loader = PyPDFLoader("documents._loader/ML.pdf")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=1000,
    chunk_overlap=200,
    
)
chunks = text_splitter.split_documents(docs)


embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

# --------------------------------------------------
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)






