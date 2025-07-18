# backend/ingest_docs.py

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Folder where your internal docs live
DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"

# Load all .pdf and .txt files
loaders = [
    DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
    DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader),
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

# Split into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Embed and store
# Option 1 (secure): set env variable OPENAI_API_KEY
# Option 2 (quick): pass the key directly here
embedding = OpenAIEmbeddings()  # or OpenAIEmbeddings(openai_api_key="sk-...")

vectordb = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)
vectordb.persist()

print("Document ingestion complete.")
