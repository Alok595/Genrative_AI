from langchain_community.document_loaders import TextLoader



data = TextLoader("documents._loader/Data_Cleaning.txt")

# Load all documents
documents = data.load()

print(documents)

# # For large datasets, lazily load documents
# for document in data.lazy_load():
#     print(document)