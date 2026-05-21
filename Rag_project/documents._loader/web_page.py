from langchain_community.document_loaders import WebBaseLoader



url="https://www.samsung.com/in/multistore/in_corporate/prelogin/?cid=in_pd_search_bing_EPP_sustain_b2b-q1-2026_text_all_1ur-464708l&utm_source=bing&utm_medium=paid&utm_campaign=-523827562-1322715246100761-kwd-82671252743746:loc-90&utm_term"


data = WebBaseLoader(url)

# Load all documents
documents = data.load()

print(documents[0].page_content)

