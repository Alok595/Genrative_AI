from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader





spliter = CharacterTextSplitter(
  separator="",
  chunk_size = 10,
  chunk_overlap=1
)

data = TextLoader("documnet_spliter/notes.txt")
documents = data.load()

chunks = spliter.split_documents(documents)
print(chunks)


for i in chunks:
  print(i.page_content)