# ingest.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os

# -----------------------------
# Config
# -----------------------------
DOCS_DIR = "docs"  # Folder with PDF files
DB_DIR = "db"      # Folder to store vectorstore

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# Collect all PDF files
all_docs = []
for file_name in os.listdir(DOCS_DIR):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCS_DIR, file_name))
        all_docs.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(all_docs)

# Create or load vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=DB_DIR
)

vectorstore.persist()
print(f"Documents indexed successfully. Total chunks: {len(chunks)}")
