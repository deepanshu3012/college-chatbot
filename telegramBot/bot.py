import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Try loading from the root .env first
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
# Also load local .env if it exists
load_dotenv()

# ── Config ──
INDEX_NAME = "college-chatbot"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

missing_keys = []
if not PINECONE_API_KEY:
    missing_keys.append("PINECONE_API_KEY")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")
if not TELEGRAM_BOT_TOKEN:
    missing_keys.append("TELEGRAM_BOT_TOKEN")

if missing_keys:
    print(f"⚠️ ERROR: You missed the key(s): {', '.join(missing_keys)}")
    print("Please add them to your .env file or Hugging Face variables before running the bot.")
    sys.exit(1)

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
    api_key=GROQ_API_KEY
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

# Memory storage: map user_id -> list of history dicts
user_memories = {}

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

async def ask_with_memory(user_id: int, question: str) -> str:
    # Initialize user memory if it doesn't exist
    if user_id not in user_memories:
        user_memories[user_id] = []
        
    history = user_memories[user_id]
    
    docs = await retriever.ainvoke(question)
    context = format_docs(docs)
    formatted_history = format_history(history)
    
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "history": formatted_history, "question": question})
    
    # Update memory
    user_memories[user_id].append({"user": question, "bot": answer})
    # Keep only the last 6 exchanges
    if len(user_memories[user_id]) > 6:
        user_memories[user_id].pop(0)
        
    return answer

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_memories[user.id] = []  # Clear memory on start
    await update.message.reply_text(
        f"Hello {user.first_name}! 🎓 I am the College Enquiry Assistant.\n"
        "Ask me anything about courses, fees, admissions, or the campus."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_keyboard = [
        ["What courses are offered?"],
        ["What are the admission requirements?"],
        ["What is the fee structure?"],
        ["Is hostel facility available?"]
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
    
    await update.message.reply_text(
        "Here are some examples of things you can ask me. Tap any of the buttons below to ask instantly!",
        reply_markup=markup
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_memories[user_id] = []
    await update.message.reply_text("🧹 Your conversation memory has been cleared. Let's start fresh!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    question = update.message.text
    
    # Send a "typing" action to Telegram
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        start_time = time.time()
        print(f"❓ Telegram User {user_id}: {question}")
        answer = await ask_with_memory(user_id, question)
        end_time = time.time()
        latency = round(end_time - start_time, 2)
        print(f"✅ Bot to {user_id} in {latency}s: {answer[:50]}...")
        await update.message.reply_text(answer)
        
        # Log latency
        log_file = os.path.join(os.path.dirname(__file__), 'latency_log.csv')
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'UserID', 'Question', 'LatencySeconds'])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, question, latency])
            
    except Exception as e:
        print(f"⚠️ Error handling message from {user_id}: {e}")
        await update.message.reply_text("⚠️ Something went wrong. Please try again later.")

def main():
    print("🚀 Starting Telegram Bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("✅ Bot is polling for messages...")
    app.run_polling()

if __name__ == '__main__':
    main()
