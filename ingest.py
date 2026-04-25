import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

print("📄 Loading your college PDF...")
loader = PyPDFLoader("data/college_info.pdf")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # increased from 500
    chunk_overlap=100  # increased from 50
)
chunks = splitter.split_documents(documents)
print(f"📦 Created {len(chunks)} chunks before deduplication")

# ── Remove duplicate chunks ──
seen = set()
unique_chunks = []
for chunk in chunks:
    text = chunk.page_content.strip()
    if text not in seen:
        seen.add(text)
        unique_chunks.append(chunk)

print(f"✅ {len(unique_chunks)} unique chunks after deduplication (removed {len(chunks) - len(unique_chunks)} duplicates)")
chunks = unique_chunks

print("🧠 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("💾 Storing chunks in ChromaDB...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db"
)
vectordb.persist()
print("✅ All done! Your college knowledge base is ready.")