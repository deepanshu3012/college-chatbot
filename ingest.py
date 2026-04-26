import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# ── Pinecone config ──
PINECONE_API_KEY = os.getenv("pcsk_b7nf9_BxgcNVXmQiJJ3t8WQrWkwo3CQCSjm2SjCuJExLSUGVycQP2ch3RLnbN8ToSxQsR")
INDEX_NAME = "college-chatbot"

print("📄 Loading your college PDF...")
loader = PyPDFLoader("data/college_info.pdf")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
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

print("☁️  Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# ── Create index if it doesn't exist ──
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"📌 Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,       # all-MiniLM-L6-v2 produces 384-dim vectors
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Index '{INDEX_NAME}' created!")
else:
    print(f"✅ Index '{INDEX_NAME}' already exists — clearing old data...")
    pc.Index(INDEX_NAME).delete(delete_all=True)
    print("✅ Old data cleared!")

print("💾 Uploading chunks to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    pinecone_api_key=PINECONE_API_KEY
)
print(f"✅ All done! {len(chunks)} chunks uploaded to Pinecone cloud.")
print(f"🌐 Your knowledge base is now live in the cloud!")