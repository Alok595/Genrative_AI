from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader




data = PyPDFLoader("documnet_spliter/Data_Cleaning.pdf")

# Load all documents
documents = data.load()

  

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
    
)


texts = text_splitter.split_documents(documents)
print(texts[0])
print(texts[1])