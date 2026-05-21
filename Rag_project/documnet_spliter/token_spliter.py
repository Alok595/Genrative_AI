from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader




data = PyPDFLoader("documnet_spliter/Data_Cleaning.pdf")

# Load all documents
documents = data.load()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

chunks = text_splitter.split_documents(documents)
print(chunks[0].page_content)