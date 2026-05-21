Here’s a **clean, corrected, and properly commented version** of your code + explanations so you can reuse it like a mini reference guide.

---

# 📄 LangChain Document Loaders (Proper Notes + Fixed Code)

```python
# =========================
# 📌 1. PDF LOADER
# =========================

from langchain_community.document_loaders import PyPDFLoader

# Load a PDF file from local path
data = PyPDFLoader("documents._loader/Data_Cleaning.pdf")

# Convert PDF into list of Document objects (page-wise)
documents = data.load()

# Each page becomes a separate Document object
# You can access specific pages like this:
print(documents[8])   # 9th page (index starts from 0)

# -----------------------------------------
# 💡 What happens internally?
# -----------------------------------------
# PDF → split into pages → each page stored as:
# Document(page_content="text", metadata={"page": 8})


# =========================
# 📌 2. TEXT FILE LOADER
# =========================

from langchain_community.document_loaders import TextLoader

# Load plain text file
data = TextLoader("documents._loader/Data_Cleaning.txt")

# Convert full text into Document object(s)
documents = data.load()

# Print full document content
print(documents[0].page_content)

# -----------------------------------------
# 💡 What happens internally?
# -----------------------------------------
# TXT file → single Document object (usually 1 big chunk)


# =========================
# 📌 3. WEB PAGE LOADER
# =========================

from langchain_community.document_loaders import WebBaseLoader

# URL to scrape content from
url = "https://www.samsung.com/in/multistore/in_corporate/prelogin/?cid=in_pd_search_bing_EPP_sustain_b2b-q1-2026_text_all_1ur-464708l&utm_source=bing&utm_medium=paid&utm_campaign=-523827562-1322715246100761-kwd-82671252743746:loc-90&utm_term"

# Load webpage content
data = WebBaseLoader(url)

documents = data.load()

# Print extracted page content
print(documents[0].page_content)

# Print full Document object (text + metadata)
print(documents)

# -----------------------------------------
# 💡 What happens internally?
# -----------------------------------------
# Web page → HTML scraped → cleaned text extracted → Document object created


# =========================
# 📌 4. LAZY LOADING (IMPORTANT FOR LARGE DATA)
# =========================

# Instead of loading everything into memory at once,
# we can load documents one by one (efficient for large datasets)

# for document in data.lazy_load():
#     print(document.page_content)

# -----------------------------------------
# 💡 Why use lazy loading?
# -----------------------------------------
# ✔ Saves memory
# ✔ Faster for huge PDFs / websites
# ✔ Streams documents one at a time
```

---

# 🧠 KEY UNDERSTANDING (VERY IMPORTANT)

## 📦 All loaders in LangChain convert data into:

```text
Document object
```

Which contains:

```python
Document(
    page_content="actual text here",
    metadata={"source": "...", "page": 1}
)
```

---

# 🔥 DIFFERENCE BETWEEN LOADERS

| Loader        | Input Type | Output Style        | Best For          |
| ------------- | ---------- | ------------------- | ----------------- |
| PyPDFLoader   | PDF file   | Page-wise Documents | PDFs, books       |
| TextLoader    | TXT file   | Single document     | Simple text files |
| WebBaseLoader | URL        | Web scraped text    | Websites          |

---

# 🚀 REAL PIPELINE (HOW IT IS USED IN RAG)

```text
Data Source (PDF / TXT / Web)
        ↓
Document Loader
        ↓
Text Splitter
        ↓
Embeddings Model
        ↓
Vector Database
        ↓
LLM (Chatbot / QA System)
```

---

# ⚠️ COMMON MISTAKES (FIXED IN YOUR CODE)

### ❌ Wrong:

```python
documents = data.load()from langchain_community.document_loaders import WebBaseLoader
```

### ✅ Correct:

Separate statements properly.

---

# 🎯 FINAL SUMMARY

* Loaders convert data → Document objects
* PDF = page-wise chunks
* TXT = single text block
* Web = scraped HTML text
* Lazy loading = memory efficient

---

If you want next upgrade, I can also make:

✅ Full **RAG chatbot project using these loaders**
✅ Add **diagram of loader → splitter → vector DB flow**
✅ Or convert this into a **GitHub-ready project structure**
