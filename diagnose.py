from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Test with a question that the bot fails to answer
query = "what is the name of college in etawah campus?"   # change this to a failing question
docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} chunks:\n")
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()