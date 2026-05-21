from langchain_community.document_loaders import PyPDFLoader




data = PyPDFLoader("documents._loader/Data_Cleaning.pdf")

# Load all documents
documents = data.load()

print(documents[8])

