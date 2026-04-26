import os
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
from datetime import datetime
from collections import Counter
import uvicorn

load_dotenv()

# ── Config ──
ADMIN_PASSWORD  = "admin123"
CHAT_LOG_FILE   = "chat_log.json"
FEEDBACK_FILE   = "feedback.json"
INDEX_NAME      = "college-chatbot"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("🔄 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("☁️  Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

print("🤖 Connecting to Groq LLM...")
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""You are a helpful and friendly college enquiry assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information, please contact the college directly."
Keep your answers clear and concise.

IMPORTANT LANGUAGE RULE:
- Detect the language of the "Current Question" below.
- If the question is in Hindi (or contains Hindi/Devanagari words), respond FULLY in Hindi.
- If the question is in English, respond in English.
- Never mix languages in a single response.
- If responding in Hindi, also translate the "I don't have that information" message to Hindi.

Context:
{context}

Conversation History:
{history}

Current Question:
{question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    if not history:
        return "No previous conversation."
    lines = []
    for h in history:
        lines.append(f"Student: {h['user']}")
        lines.append(f"Assistant: {h['bot']}")
    return "\n".join(lines)

def ask_with_memory(question: str, history: list) -> str:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    formatted_history = format_history(history)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "history": formatted_history, "question": question})

def rebuild_knowledge_base(pdf_path: str):
    global vectorstore, retriever
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    seen, unique_chunks = set(), []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)

    # ── Clear existing Pinecone index and re-upload ──
    try:
        pc.Index(INDEX_NAME).delete(delete_all=True)
        print("✅ Old Pinecone data cleared")
    except Exception as e:
        print(f"⚠️ Could not clear index: {e}")

    vectorstore = PineconeVectorStore.from_documents(
        documents=unique_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print(f"✅ Knowledge base rebuilt with {len(unique_chunks)} chunks in Pinecone")

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_chat_log():
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_log(data):
    with open(CHAT_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

app = FastAPI()

class Message(BaseModel):
    message: str
    history: List[dict] = []

class Feedback(BaseModel):
    question: str
    answer: str
    rating: str

@app.post("/ask")
async def ask(payload: Message):
    if not payload.message.strip():
        return {"answer": "Please ask a question! / कृपया एक प्रश्न पूछें!"}
    print(f"❓ Question: {payload.message}")
    answer = ask_with_memory(payload.message, payload.history)
    print(f"✅ Answer: {answer}")
    log = load_chat_log()
    log.append({"question": payload.message, "answer": answer, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    save_chat_log(log)
    return {"answer": answer}

@app.post("/feedback")
async def feedback(payload: Feedback):
    data = load_feedback()
    data.append({"question": payload.question, "answer": payload.answer, "rating": payload.rating, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    save_feedback(data)
    print(f"{'👍' if payload.rating == 'up' else '👎'} Feedback: {payload.question[:50]}")
    return {"status": "saved"}

@app.get("/admin", response_class=HTMLResponse)
async def admin_login():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Admin Login</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#f0faf4;font-family:'Plus Jakarta Sans',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}
  .card{background:white;border:1.5px solid #bbf7d0;border-radius:24px;padding:2.5rem 2rem;width:100%;max-width:380px;box-shadow:0 4px 24px rgba(34,197,94,.1);text-align:center}
  .icon{font-size:2.5rem;margin-bottom:1rem}
  h1{font-size:1.4rem;font-weight:800;color:#0a2e1a;margin-bottom:.3rem}
  p{font-size:.8rem;color:#6b7280;margin-bottom:1.5rem}
  input{width:100%;border:1.5px solid #bbf7d0;border-radius:12px;padding:.7rem 1rem;font-family:'Plus Jakarta Sans',sans-serif;font-size:.9rem;color:#14532d;outline:none;margin-bottom:1rem;transition:border .2s}
  input:focus{border-color:#22c55e}
  button{width:100%;background:#16a34a;border:none;border-radius:12px;color:white;font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:.9rem;padding:.75rem;cursor:pointer;transition:all .2s}
  button:hover{background:#15803d;transform:translateY(-1px)}
  .back{display:inline-block;margin-top:1rem;font-size:.78rem;color:#16a34a;text-decoration:none;font-weight:600}
</style>
</head>
<body>
<div class="card">
  <div class="icon">🔐</div>
  <h1>Admin Panel</h1>
  <p>Enter your admin password to continue</p>
  <form method="POST" action="/admin/login">
    <input type="password" name="password" placeholder="Enter admin password" required autofocus/>
    <button type="submit">Login →</button>
  </form>
  <a href="/" class="back">← Back to Chatbot</a>
</div>
</body>
</html>""")

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login_post(request: Request, password: str = Form(...)):
    if password != ADMIN_PASSWORD:
        return HTMLResponse(content="""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<style>*{box-sizing:border-box;margin:0;padding:0}body{background:#f0faf4;font-family:sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}.card{background:white;border:1.5px solid #fecaca;border-radius:24px;padding:2.5rem 2rem;width:100%;max-width:380px;text-align:center}.icon{font-size:2.5rem;margin-bottom:1rem}h1{color:#dc2626;font-size:1.2rem;margin-bottom:1rem}a{display:inline-block;background:#16a34a;color:white;padding:.6rem 1.4rem;border-radius:10px;text-decoration:none;font-weight:700;font-size:.85rem}</style>
</head><body><div class="card"><div class="icon">❌</div><h1>Incorrect Password</h1><a href="/admin">Try Again</a></div></body></html>""")
    return RedirectResponse(url=f"/admin/panel?pwd={password}", status_code=303)

@app.get("/admin/panel", response_class=HTMLResponse)
async def admin_panel(pwd: str = ""):
    if pwd != ADMIN_PASSWORD:
        return RedirectResponse(url="/admin")

    feedback_data = load_feedback()
    chat_data = load_chat_log()
    total_chats = len(chat_data)
    thumbs_up = sum(1 for d in feedback_data if d["rating"] == "up")
    thumbs_down = sum(1 for d in feedback_data if d["rating"] == "down")
    total_fb = len(feedback_data)
    satisfaction = round((thumbs_up / total_fb * 100) if total_fb > 0 else 0)

    recent_chats = chat_data[-8:][::-1]
    chat_rows = ""
    for d in recent_chats:
        q = d['question'][:55] + ('...' if len(d['question']) > 55 else '')
        a = d['answer'][:80] + ('...' if len(d['answer']) > 80 else '')
        chat_rows += f"<tr><td>{d['timestamp']}</td><td>{q}</td><td>{a}</td></tr>"

    chat_table = f"<table><thead><tr><th>Time</th><th>Question</th><th>Answer</th></tr></thead><tbody>{chat_rows}</tbody></table>" if chat_rows else "<div class='empty'>No chats yet!</div>"

    # ── Pinecone status ──
    try:
        index_info = pc.Index(INDEX_NAME).describe_index_stats()
        total_vectors = index_info.get("total_vector_count", 0)
        db_status = f"✅ Pinecone connected — {total_vectors} vectors"
    except Exception:
        db_status = "⚠️ Pinecone connection issue"

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Admin Panel</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#f0faf4;font-family:'Plus Jakarta Sans',sans-serif;padding:2rem 1rem}}
  .header{{background:#0a2e1a;color:#4ade80;padding:1.5rem 2rem;border-radius:16px;margin-bottom:1.5rem;display:flex;justify-content:space-between;align-items:center}}
  .header h1{{font-size:1.4rem;font-weight:800}}
  .header-links{{display:flex;gap:.6rem}}
  .hlink{{background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.3);border-radius:8px;color:#4ade80;font-size:.72rem;font-weight:700;padding:.35rem .8rem;text-decoration:none}}
  .hlink:hover{{background:rgba(74,222,128,.25)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem}}
  .stat{{background:white;border:1.5px solid #bbf7d0;border-radius:16px;padding:1.2rem;text-align:center;box-shadow:0 2px 12px rgba(34,197,94,.08)}}
  .stat .val{{font-size:2rem;font-weight:800;color:#16a34a}}
  .stat .lbl{{font-size:.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:.3rem}}
  .section{{background:white;border:1.5px solid #bbf7d0;border-radius:16px;padding:1.4rem;margin-bottom:1.5rem}}
  .section h2{{font-size:.85rem;font-weight:700;color:#16a34a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem}}
  .status-row{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem}}
  .status-badge{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:.4rem .8rem;font-size:.8rem;color:#14532d;font-weight:600}}
  .btn{{border:none;border-radius:12px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:.85rem;padding:.65rem 1.3rem;cursor:pointer;transition:all .2s;white-space:nowrap}}
  .btn-green{{background:#16a34a;color:white;box-shadow:0 2px 10px rgba(22,163,74,.3)}}
  .btn-green:hover{{background:#15803d;transform:translateY(-1px)}}
  .btn-red{{background:white;border:1.5px solid #fecaca;color:#dc2626}}
  .btn-red:hover{{background:#fee2e2}}
  .btn-row{{display:flex;gap:.7rem;flex-wrap:wrap}}
  table{{width:100%;border-collapse:collapse;font-size:.82rem}}
  th{{background:#f0fdf4;color:#16a34a;font-weight:700;padding:.6rem .8rem;text-align:left;font-size:.72rem;text-transform:uppercase}}
  td{{padding:.55rem .8rem;border-top:1px solid #f0fdf4;color:#374151;vertical-align:top;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .empty{{text-align:center;color:#9ca3af;padding:2rem;font-size:.88rem}}
  .alert{{background:#dcfce7;border:1px solid #86efac;border-radius:10px;padding:.8rem 1rem;font-size:.82rem;color:#14532d;font-weight:600;display:none;margin-top:.5rem}}
  .alert.show{{display:block}}
</style>
</head>
<body>
<div class="header">
  <h1>🛠️ Admin Panel</h1>
  <div class="header-links">
    <a href="/dashboard" class="hlink">📊 Dashboard</a>
    <a href="/" class="hlink">← Chatbot</a>
  </div>
</div>
<div class="grid">
  <div class="stat"><div class="val">{total_chats}</div><div class="lbl">Total Chats</div></div>
  <div class="stat"><div class="val" style="color:#16a34a">👍 {thumbs_up}</div><div class="lbl">Helpful</div></div>
  <div class="stat"><div class="val" style="color:#dc2626">👎 {thumbs_down}</div><div class="lbl">Not Helpful</div></div>
  <div class="stat"><div class="val">{satisfaction}%</div><div class="lbl">Satisfaction</div></div>
</div>
<div class="section">
  <h2>📚 Knowledge Base Management</h2>
  <div class="status-row">
    <div class="status-badge">☁️ Pinecone Cloud DB</div>
    <div class="status-badge">{db_status}</div>
  </div>
  <p style="font-size:.82rem;color:#6b7280;margin-bottom:.8rem">Upload a new college PDF to update the Pinecone knowledge base in the cloud.</p>
  <form method="POST" action="/admin/upload?pwd={pwd}" enctype="multipart/form-data">
    <input type="file" name="file" accept=".pdf" required style="border:2px dashed #bbf7d0;border-radius:12px;padding:1rem;width:100%;font-size:.85rem;color:#14532d;background:#f0fdf4;cursor:pointer;margin-bottom:.8rem"/>
    <div class="btn-row">
      <button type="submit" class="btn btn-green">📤 Upload & Rebuild Knowledge Base</button>
    </div>
  </form>
  <div class="alert" id="upload-alert">✅ PDF uploaded and Pinecone knowledge base rebuilt successfully!</div>
</div>
<div class="section">
  <h2>🗂️ Data Management</h2>
  <p style="font-size:.82rem;color:#6b7280;margin-bottom:1rem">Manage stored feedback and chat logs. These actions cannot be undone.</p>
  <div class="btn-row">
    <form method="POST" action="/admin/clear-feedback?pwd={pwd}" onsubmit="return confirm('Clear all feedback data?')">
      <button type="submit" class="btn btn-red">🗑️ Clear Feedback Data</button>
    </form>
    <form method="POST" action="/admin/clear-chats?pwd={pwd}" onsubmit="return confirm('Clear all chat logs?')">
      <button type="submit" class="btn btn-red">🗑️ Clear Chat Logs</button>
    </form>
  </div>
</div>
<div class="section">
  <h2>💬 Recent Chat Logs</h2>
  {chat_table}
</div>
<script>
  if(window.location.search.includes('success=1')){{
    const a=document.getElementById('upload-alert');a.classList.add('show');
    setTimeout(()=>a.classList.remove('show'),4000);
  }}
</script>
</body>
</html>""")

@app.post("/admin/upload")
async def admin_upload(pwd: str = "", file: UploadFile = File(default=None)):
    if pwd != ADMIN_PASSWORD:
        return RedirectResponse(url="/admin")
    if file is None or file.filename == "":
        return RedirectResponse(url=f"/admin/panel?pwd={pwd}&error=1", status_code=303)
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/college_info.pdf"
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)
    print(f"📄 New PDF uploaded: {file.filename}")
    rebuild_knowledge_base(pdf_path)
    return RedirectResponse(url=f"/admin/panel?pwd={pwd}&success=1", status_code=303)

@app.post("/admin/clear-feedback")
async def clear_feedback(pwd: str = ""):
    if pwd != ADMIN_PASSWORD:
        return RedirectResponse(url="/admin")
    save_feedback([])
    return RedirectResponse(url=f"/admin/panel?pwd={pwd}", status_code=303)

@app.post("/admin/clear-chats")
async def clear_chats(pwd: str = ""):
    if pwd != ADMIN_PASSWORD:
        return RedirectResponse(url="/admin")
    save_chat_log([])
    return RedirectResponse(url=f"/admin/panel?pwd={pwd}", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    data = load_feedback()
    total = len(data)
    thumbs_up = sum(1 for d in data if d["rating"] == "up")
    thumbs_down = sum(1 for d in data if d["rating"] == "down")
    satisfaction = round((thumbs_up / total * 100) if total > 0 else 0)
    sat_color = "#16a34a" if satisfaction >= 70 else "#f59e0b" if satisfaction >= 40 else "#dc2626"
    recent = data[-10:][::-1]
    rows = ""
    for d in recent:
        emoji = "👍" if d["rating"] == "up" else "👎"
        color = "#16a34a" if d["rating"] == "up" else "#dc2626"
        q = d['question'][:60] + ('...' if len(d['question']) > 60 else '')
        rows += f"<tr><td>{d['timestamp']}</td><td>{q}</td><td style='color:{color};font-size:1.2rem'>{emoji}</td></tr>"
    q_counts = Counter(d["question"] for d in data)
    top_questions = q_counts.most_common(5)
    top_q_rows = ""
    for q, count in top_questions:
        top_q_rows += f"<li><span>{q[:55]}{'...' if len(q)>55 else ''}</span><b>{count}x</b></li>"
    table_html = f"<table><thead><tr><th>Time</th><th>Question</th><th>Rating</th></tr></thead><tbody>{rows}</tbody></table>" if rows else "<div class='empty'>No feedback yet!</div>"
    top_html = f"<ul class='top-q-list'>{top_q_rows}</ul>" if top_q_rows else "<div class='empty'>No data yet!</div>"
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}body{{background:#f0faf4;font-family:'Plus Jakarta Sans',sans-serif;padding:2rem 1rem}}
  .header{{background:#0a2e1a;color:#4ade80;text-align:center;padding:1.5rem;border-radius:16px;margin-bottom:1.5rem}}
  .header h1{{font-size:1.6rem;font-weight:800}}.header p{{color:#34d399;font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;margin-top:.3rem}}
  .back-btn{{display:inline-block;margin-bottom:1rem;background:white;border:1.5px solid #bbf7d0;border-radius:10px;color:#16a34a;font-weight:700;font-size:.82rem;padding:.4rem 1rem;text-decoration:none}}
  .back-btn:hover{{background:#16a34a;color:white}}
  .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-bottom:1.5rem}}
  .stat-card{{background:white;border:1.5px solid #bbf7d0;border-radius:16px;padding:1.2rem;text-align:center;box-shadow:0 2px 12px rgba(34,197,94,.08)}}
  .stat-card .value{{font-size:2.2rem;font-weight:800;color:#16a34a}}.stat-card .label{{font-size:.75rem;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:.3rem}}
  .section{{background:white;border:1.5px solid #bbf7d0;border-radius:16px;padding:1.4rem;margin-bottom:1.5rem}}
  .section h2{{font-size:.85rem;font-weight:700;color:#16a34a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th{{background:#f0fdf4;color:#16a34a;font-weight:700;padding:.6rem .8rem;text-align:left;font-size:.75rem;text-transform:uppercase}}
  td{{padding:.6rem .8rem;border-top:1px solid #f0fdf4;color:#374151;vertical-align:top}}
  .top-q-list{{list-style:none;display:flex;flex-direction:column;gap:.6rem}}
  .top-q-list li{{display:flex;justify-content:space-between;align-items:center;padding:.6rem .8rem;background:#f0fdf4;border-radius:10px;font-size:.85rem}}
  .top-q-list li b{{color:#16a34a;font-weight:800;white-space:nowrap;margin-left:.5rem}}
  .empty{{text-align:center;color:#9ca3af;padding:2rem;font-size:.9rem}}
  .progress-bar{{background:#dcfce7;border-radius:99px;height:10px;margin-top:.5rem;overflow:hidden}}
  .progress-fill{{background:{sat_color};height:100%;border-radius:99px;width:{satisfaction}%}}
</style></head><body>
<a href="/" class="back-btn">← Back to Chatbot</a>
<div class="header"><h1>📊 Analytics Dashboard</h1><p>College Enquiry Assistant — Live Feedback Stats</p></div>
<div class="stats-grid">
  <div class="stat-card"><div class="value">{total}</div><div class="label">Total Responses</div></div>
  <div class="stat-card"><div class="value" style="color:#16a34a">👍 {thumbs_up}</div><div class="label">Helpful</div></div>
  <div class="stat-card"><div class="value" style="color:#dc2626">👎 {thumbs_down}</div><div class="label">Not Helpful</div></div>
  <div class="stat-card"><div class="value" style="color:{sat_color}">{satisfaction}%</div><div class="label">Satisfaction</div><div class="progress-bar"><div class="progress-fill"></div></div></div>
</div>
<div class="section"><h2>🔥 Most Asked Questions</h2>{top_html}</div>
<div class="section"><h2>🕐 Recent Feedback</h2>{table_html}</div>
</body></html>""")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>College Enquiry Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:#f0faf4;font-family:'Plus Jakarta Sans',sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center}
  .header{width:100%;background:#0a2e1a;padding:1.8rem 2rem;text-align:center;border-radius:0 0 28px 28px;margin-bottom:1.5rem;position:relative}
  .header h1{font-size:2rem;font-weight:800;color:#4ade80;letter-spacing:-.02em;margin-bottom:.3rem}
  .header p{color:#34d399;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;opacity:.85}
  .header-links{position:absolute;top:1rem;right:1.2rem;display:flex;gap:.5rem}
  .hlink{background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.3);border-radius:8px;color:#4ade80;font-size:.68rem;font-weight:700;padding:.3rem .7rem;text-decoration:none;letter-spacing:.05em;transition:all .2s}
  .hlink:hover{background:rgba(74,222,128,.25)}
  .container{width:100%;max-width:780px;padding:0 1rem 2rem;display:flex;flex-direction:column;gap:1rem}
  .chat-window{background:white;border:1.5px solid #bbf7d0;border-radius:20px;height:460px;overflow-y:auto;padding:1.2rem;display:flex;flex-direction:column;gap:1rem;box-shadow:0 4px 24px rgba(34,197,94,.08)}
  .chat-window::-webkit-scrollbar{width:5px}.chat-window::-webkit-scrollbar-track{background:#f0fdf4}.chat-window::-webkit-scrollbar-thumb{background:#86efac;border-radius:4px}
  .empty-state{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.6rem}
  .empty-state .icon{font-size:2.5rem}
  .empty-state p{font-size:.9rem;font-weight:500;color:#4ade80}
  .msg-row{display:flex;align-items:flex-end;gap:.6rem}.msg-row.user{flex-direction:row-reverse}
  .avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0}
  .avatar.bot{background:#0a2e1a}.avatar.user{background:#16a34a}
  .bubble-wrap{display:flex;flex-direction:column;gap:.3rem;max-width:75%}.msg-row.user .bubble-wrap{align-items:flex-end}
  .bubble{padding:.8rem 1.1rem;font-size:.93rem;line-height:1.7}
  .bubble.user{background:#0a2e1a;color:white;border-radius:18px 18px 4px 18px}
  .bubble.bot{background:#f0fdf4;color:#14532d;border-radius:18px 18px 18px 4px;border-left:3px solid #22c55e}
  .feedback-row{display:flex;gap:.4rem;padding-left:.3rem;align-items:center}
  .fb-btn{background:white;border:1px solid #bbf7d0;border-radius:8px;font-size:.85rem;padding:.2rem .6rem;cursor:pointer;transition:all .2s;color:#6b7280}
  .fb-btn:hover{transform:scale(1.15)}.fb-btn.up.selected{background:#dcfce7;border-color:#16a34a}.fb-btn.down.selected{background:#fee2e2;border-color:#dc2626}
  .fb-thanks{font-size:.72rem;color:#16a34a;font-weight:600;display:none}
  .typing-bubble{background:#f0fdf4;border-radius:18px 18px 18px 4px;border-left:3px solid #22c55e;padding:.8rem 1.2rem;display:flex;gap:5px;align-items:center}
  .dot{width:7px;height:7px;background:#22c55e;border-radius:50%;animation:bounce 1.2s infinite}
  .dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}
  .input-area{background:white;border:1.5px solid #22c55e;border-radius:16px;padding:.5rem .5rem .5rem 1rem;display:flex;align-items:center;gap:.6rem;box-shadow:0 2px 16px rgba(34,197,94,.1)}
  .input-area textarea{flex:1;border:none;outline:none;background:transparent;color:#14532d;font-family:'Plus Jakarta Sans',sans-serif;font-size:.95rem;resize:none;height:44px;padding-top:.6rem;line-height:1.5}
  .input-area textarea::placeholder{color:#4ade80;font-weight:500}
  .btn-send{background:#16a34a;border:none;border-radius:12px;color:white;font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:.88rem;padding:.65rem 1.3rem;cursor:pointer;transition:all .2s;box-shadow:0 2px 10px rgba(22,163,74,.3);white-space:nowrap}
  .btn-send:hover{background:#15803d;transform:translateY(-1px)}.btn-send:disabled{background:#86efac;cursor:not-allowed;transform:none}
  .btn-clear{background:white;border:1.5px solid #bbf7d0;border-radius:12px;color:#16a34a;font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:.85rem;padding:.65rem 1rem;cursor:pointer;transition:all .2s}
  .btn-clear:hover{border-color:#f87171;color:#dc2626;background:#fff5f5}
  .btn-voice{background:white;border:1.5px solid #bbf7d0;border-radius:12px;font-size:1.1rem;padding:.65rem .9rem;cursor:pointer;transition:all .2s;white-space:nowrap;line-height:1}
  .btn-voice:hover{background:#f0fdf4;border-color:#22c55e}
  .btn-voice.listening{background:#16a34a;border-color:#16a34a;animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(22,163,74,.4)}50%{box-shadow:0 0 0 8px rgba(22,163,74,0)}}
  .voice-status{display:none;text-align:center;font-size:.78rem;color:#16a34a;font-weight:600;letter-spacing:.05em;padding:.3rem 0}
  .voice-status.visible{display:block}
  .bottom-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
  .speaker-row{display:flex;align-items:center;gap:.5rem;font-size:.78rem;color:#4ade80;font-weight:600;letter-spacing:.05em;text-transform:uppercase}
  .toggle-switch{position:relative;width:36px;height:20px}.toggle-switch input{opacity:0;width:0;height:0}
  .toggle-slider{position:absolute;cursor:pointer;inset:0;background:#bbf7d0;border-radius:20px;transition:.3s}
  .toggle-slider:before{content:"";position:absolute;width:14px;height:14px;left:3px;bottom:3px;background:white;border-radius:50%;transition:.3s}
  input:checked+.toggle-slider{background:#16a34a}input:checked+.toggle-slider:before{transform:translateX(16px)}
  .memory-badge{font-size:.72rem;color:#16a34a;font-weight:700;background:#dcfce7;border:1px solid #bbf7d0;border-radius:50px;padding:.25rem .7rem;letter-spacing:.04em}
  .lang-row{display:flex;align-items:center;gap:.5rem}
  .lang-btn{background:white;border:1.5px solid #bbf7d0;border-radius:8px;font-size:.78rem;font-weight:700;padding:.3rem .7rem;cursor:pointer;transition:all .2s;color:#6b7280}
  .lang-btn.active{background:#16a34a;border-color:#16a34a;color:white}
  .examples-label{color:#16a34a;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem}
  .examples-row{display:flex;flex-wrap:wrap;gap:.5rem}
  .chip{background:white;border:1.5px solid #bbf7d0;border-radius:50px;color:#15803d;font-family:'Plus Jakarta Sans',sans-serif;font-size:.8rem;font-weight:600;padding:.4rem 1rem;cursor:pointer;transition:all .2s}
  .chip:hover{background:#16a34a;border-color:#16a34a;color:white;transform:translateY(-1px)}
</style>
</head>
<body>
<div class="header">
  <h1>🎓 College Enquiry Assistant</h1>
  <p>Powered by Groq &nbsp;·&nbsp; LLaMA 3.3 &nbsp;·&nbsp; RAG</p>
  <div class="header-links">
    <a href="/dashboard" class="hlink">📊 Dashboard</a>
    <a href="/admin" class="hlink">🛠️ Admin</a>
  </div>
</div>
<div class="container">
  <div class="chat-window" id="chat-window">
    <div class="empty-state" id="empty-state">
      <div class="icon">🎓</div>
      <p id="empty-text">Ask me anything about the college</p>
    </div>
  </div>
  <div class="voice-status" id="voice-status">🎤 Listening... / सुन रहा हूँ...</div>
  <div class="input-area">
    <textarea id="msg-input" placeholder="Ask about courses, fees, admissions..." onkeydown="handleKey(event)"></textarea>
    <button class="btn-voice" id="voice-btn" onclick="toggleVoice()" title="Click to speak">🎤</button>
    <button class="btn-send" id="send-btn" onclick="sendMessage()">Send ➤</button>
    <button class="btn-clear" onclick="clearChat()">✕</button>
  </div>
  <div class="bottom-row">
    <div class="speaker-row">
      <label class="toggle-switch"><input type="checkbox" id="speaker-toggle" checked><span class="toggle-slider"></span></label>
      🔊 Read aloud
    </div>
    <div class="lang-row">
      <span style="font-size:.72rem;color:#4ade80;font-weight:700;text-transform:uppercase;letter-spacing:.05em">🌐 Lang:</span>
      <button class="lang-btn active" id="btn-en" onclick="setLang('en')">EN</button>
      <button class="lang-btn" id="btn-hi" onclick="setLang('hi')">हि</button>
    </div>
    <div class="memory-badge" id="memory-badge">🧠 Memory: 0 exchanges</div>
  </div>
  <div>
    <div class="examples-label" id="examples-label">✦ Quick questions</div>
    <div class="examples-row" id="examples-row">
      <button class="chip" onclick="useExample(this)">What courses are offered?</button>
      <button class="chip" onclick="useExample(this)">What are the admission requirements?</button>
      <button class="chip" onclick="useExample(this)">What is the fee structure?</button>
      <button class="chip" onclick="useExample(this)">Is hostel facility available?</button>
      <button class="chip" onclick="useExample(this)">What is the college address?</button>
    </div>
  </div>
</div>
<script>
  const chatWindow=document.getElementById('chat-window');
  const msgInput=document.getElementById('msg-input');
  const sendBtn=document.getElementById('send-btn');
  const emptyState=document.getElementById('empty-state');
  const voiceStatus=document.getElementById('voice-status');
  const speakerToggle=document.getElementById('speaker-toggle');
  const memoryBadge=document.getElementById('memory-badge');
  let conversationHistory=[],lastQuestion='',currentLang='en';

  const langConfig={
    en:{placeholder:'Ask about courses, fees, admissions, hostel...',emptyText:'Ask me anything about the college',examplesLabel:'✦ Quick questions',voiceStatus:'🎤 Listening... speak your question',chips:['What courses are offered?','What are the admission requirements?','What is the fee structure?','Is hostel facility available?','What is the college address?'],speechLang:'en-IN',memoryText:(n)=>`🧠 Memory: ${n} exchange${n!==1?'s':''}`},
    hi:{placeholder:'कोर्स, फीस, एडमिशन के बारे में पूछें...',emptyText:'कॉलेज के बारे में कुछ भी पूछें',examplesLabel:'✦ जल्दी सवाल',voiceStatus:'🎤 सुन रहा हूँ... बोलें',chips:['कौन से कोर्स उपलब्ध हैं?','एडमिशन की योग्यता क्या है?','फीस कितनी है?','क्या हॉस्टल की सुविधा है?','कॉलेज का पता क्या है?'],speechLang:'hi-IN',memoryText:(n)=>`🧠 याददाश्त: ${n} बातचीत`}
  };

  function setLang(lang){
    currentLang=lang;const cfg=langConfig[lang];
    document.getElementById('btn-en').classList.toggle('active',lang==='en');
    document.getElementById('btn-hi').classList.toggle('active',lang==='hi');
    msgInput.placeholder=cfg.placeholder;
    document.getElementById('empty-text').textContent=cfg.emptyText;
    document.getElementById('examples-label').textContent=cfg.examplesLabel;
    voiceStatus.textContent=cfg.voiceStatus;
    const row=document.getElementById('examples-row');row.innerHTML='';
    cfg.chips.forEach(c=>{const btn=document.createElement('button');btn.className='chip';btn.textContent=c;btn.onclick=()=>useExample(btn);row.appendChild(btn);});
    if(recognition)recognition.lang=cfg.speechLang;
    updateMemoryBadge();
  }

  function updateMemoryBadge(){const n=conversationHistory.length;memoryBadge.textContent=langConfig[currentLang].memoryText(n);}
  function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}}
  function useExample(btn){msgInput.value=btn.textContent;msgInput.focus();}
  function clearChat(){
    chatWindow.innerHTML='';chatWindow.appendChild(emptyState);emptyState.style.display='flex';
    conversationHistory=[];lastQuestion='';updateMemoryBadge();
    window.speechSynthesis&&window.speechSynthesis.cancel();
  }
  function appendMessage(role,text){
    emptyState.style.display='none';
    const row=document.createElement('div');row.className=`msg-row ${role}`;
    const avatar=document.createElement('div');avatar.className=`avatar ${role}`;avatar.textContent=role==='bot'?'🎓':'🧑';
    const wrap=document.createElement('div');wrap.className='bubble-wrap';
    const bubble=document.createElement('div');bubble.className=`bubble ${role}`;bubble.textContent=text;
    wrap.appendChild(bubble);
    if(role==='bot'){
      const fbRow=document.createElement('div');fbRow.className='feedback-row';
      const upBtn=document.createElement('button');upBtn.className='fb-btn up';upBtn.textContent='👍';
      const downBtn=document.createElement('button');downBtn.className='fb-btn down';downBtn.textContent='👎';
      const thanks=document.createElement('span');thanks.className='fb-thanks';thanks.textContent=currentLang==='hi'?'धन्यवाद!':'Thanks!';
      const q=lastQuestion;
      upBtn.onclick=()=>submitFeedback(q,text,'up',upBtn,downBtn,thanks);
      downBtn.onclick=()=>submitFeedback(q,text,'down',upBtn,downBtn,thanks);
      fbRow.appendChild(upBtn);fbRow.appendChild(downBtn);fbRow.appendChild(thanks);wrap.appendChild(fbRow);
    }
    row.appendChild(avatar);row.appendChild(wrap);chatWindow.appendChild(row);chatWindow.scrollTop=chatWindow.scrollHeight;
  }
  async function submitFeedback(question,answer,rating,upBtn,downBtn,thanks){
    upBtn.disabled=true;downBtn.disabled=true;
    upBtn.classList.toggle('selected',rating==='up');downBtn.classList.toggle('selected',rating==='down');
    thanks.style.display='inline';
    await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,answer,rating})});
  }
  function showTyping(){
    emptyState.style.display='none';
    const row=document.createElement('div');row.className='msg-row bot';row.id='typing-row';
    const avatar=document.createElement('div');avatar.className='avatar bot';avatar.textContent='🎓';
    const typing=document.createElement('div');typing.className='typing-bubble';
    typing.innerHTML='<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    row.appendChild(avatar);row.appendChild(typing);chatWindow.appendChild(row);chatWindow.scrollTop=chatWindow.scrollHeight;
  }
  function removeTyping(){const t=document.getElementById('typing-row');if(t)t.remove();}
  async function sendMessage(){
    const text=msgInput.value.trim();if(!text)return;
    lastQuestion=text;msgInput.value='';sendBtn.disabled=true;
    appendMessage('user',text);showTyping();
    try{
      const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,history:conversationHistory})});
      const data=await res.json();removeTyping();appendMessage('bot',data.answer);
      conversationHistory.push({user:text,bot:data.answer});
      if(conversationHistory.length>6)conversationHistory.shift();
      updateMemoryBadge();
      if(speakerToggle.checked)speakAnswer(data.answer);
    }catch(err){removeTyping();appendMessage('bot','⚠️ Something went wrong. / कुछ गलत हो गया।');}
    finally{sendBtn.disabled=false;msgInput.focus();}
  }
  const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
  let recognition=null,isListening=false;
  if(SpeechRecognition){
    recognition=new SpeechRecognition();recognition.continuous=false;recognition.interimResults=false;
    recognition.lang=langConfig[currentLang].speechLang;
    recognition.onresult=function(e){const t=e.results[0][0].transcript;msgInput.value=t;stopListening();sendMessage();};
    recognition.onerror=function(){stopListening();};recognition.onend=function(){stopListening();};
  }
  function toggleVoice(){if(!recognition){alert('Use Chrome or Edge for voice.');return;}isListening?stopListening():startListening();}
  function startListening(){isListening=true;if(recognition)recognition.lang=langConfig[currentLang].speechLang;const b=document.getElementById('voice-btn');b.classList.add('listening');b.textContent='🔴';voiceStatus.classList.add('visible');msgInput.placeholder=langConfig[currentLang].voiceStatus;recognition.start();}
  function stopListening(){isListening=false;const b=document.getElementById('voice-btn');b.classList.remove('listening');b.textContent='🎤';voiceStatus.classList.remove('visible');msgInput.placeholder=langConfig[currentLang].placeholder;try{recognition.stop();}catch(e){}}
  function speakAnswer(text){
    if(!window.speechSynthesis)return;window.speechSynthesis.cancel();
    const u=new SpeechSynthesisUtterance(text);
    const hasHindi=/[\u0900-\u097F]/.test(text);
    u.lang=hasHindi?'hi-IN':'en-IN';u.rate=0.95;u.pitch=1;
    window.speechSynthesis.speak(u);
  }
</script>
</body>
</html>""")

print("🚀 Launching at http://0.0.0.0:7860")
uvicorn.run(app, host="0.0.0.0", port=7860)