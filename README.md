\# 🎓 College Enquiry Chatbot



An AI-powered College Enquiry Chatbot built using Python, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Pinecone Vector Database. The chatbot answers student queries using college-specific documents, making information retrieval fast and accurate.



\---



\## 🚀 Features



\- 🤖 AI-powered conversational chatbot

\- 📚 Retrieval-Augmented Generation (RAG)

\- 🔍 Semantic search using Pinecone

\- 📄 Answers based on college documents

\- 💬 Gradio web interface

\- 🐳 Docker support

\- ☁️ Ready for deployment on Hugging Face Spaces and Render



\---



\## 📂 Project Structure



```

.

├── app.py                  # Main chatbot application

├── ingest.py               # Upload and index documents

├── diagnose.py             # Diagnostic utility

├── start.sh                # Startup script

├── Dockerfile              # Docker configuration

├── requirements.txt        # Python dependencies

├── .python-version         # Python version

├── .gitignore

├── .gitattributes

├── data/                   # Source documents

├── db/                     # Local database/cache

└── .gradio/                # Gradio configuration

```



\---



\## 🛠️ Technologies Used



\- Python

\- Gradio

\- LangChain

\- Pinecone

\- Hugging Face Embeddings

\- OpenAI / Groq / Gemini (depending on configuration)

\- Docker



\---



\## ⚙️ Installation



\### 1. Clone the repository



```bash

git clone https://github.com/your-username/college-enquiry-chatbot.git



cd college-enquiry-chatbot

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



Activate it



\*\*Windows\*\*



```bash

venv\\Scripts\\activate

```



\*\*Linux / macOS\*\*



```bash

source venv/bin/activate

```



\---



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\---



\## 🔑 Environment Variables



Create a `.env` file in the project root.



Example:



```env

PINECONE\_API\_KEY=your\_pinecone\_api\_key

PINECONE\_INDEX\_NAME=college-chatbot

OPENAI\_API\_KEY=your\_openai\_key

HUGGINGFACE\_API\_KEY=your\_huggingface\_key

```



Use only the variables required by your project configuration.



\---



\## 📄 Index Documents



Place your PDFs or documents inside the `data/` folder.



Run:



```bash

python ingest.py

```



This will:



\- Read the documents

\- Split them into chunks

\- Generate embeddings

\- Upload them to Pinecone



\---



\## ▶️ Run the Application



```bash

python app.py

```



or



```bash

bash start.sh

```



The Gradio interface will open in your browser.



\---



\## 🐳 Run with Docker



Build the Docker image



```bash

docker build -t college-chatbot .

```



Run the container



```bash

docker run -p 7860:7860 college-chatbot

```



\---



\## 💬 Example Questions



\- What courses are offered?

\- What is the admission process?

\- What are the eligibility criteria?

\- What are the hostel facilities?

\- What is the fee structure?

\- When is the admission deadline?

\- What scholarships are available?

\- What documents are required?



\---



\## 📌 How It Works



1\. College documents are stored in the `data/` folder.

2\. `ingest.py` processes the documents.

3\. Text is converted into embeddings.

4\. Embeddings are stored in Pinecone.

5\. User submits a query.

6\. Relevant document chunks are retrieved.

7\. The LLM generates an answer using the retrieved context.

8\. The response is displayed in the Gradio interface.



\---



\## 📦 Requirements



\- Python 3.10+

\- Pinecone Account

\- LLM API Key

\- Internet connection



\---



\## 🚀 Deployment



This project can be deployed on:



\- Hugging Face Spaces

\- Render

\- Railway

\- Docker

\- AWS

\- Azure

\- Google Cloud



\---



\## 📸 Screenshots



Add screenshots of the chatbot interface here.



```

docs/

├── home.png

├── chat.png

└── response.png

```



\---



\## 🤝 Contributing



Contributions are welcome!



1\. Fork the repository.

2\. Create a new branch.

3\. Commit your changes.

4\. Push the branch.

5\. Open a Pull Request.



\---



\## 📝 License



This project is licensed under the MIT License.



\---



\## 👨‍💻 Author



\*\*Sachin Maurya\*\*



GitHub: https://github.com/your-username



\---



\## ⭐ Support



If you found this project useful, consider giving it a ⭐ on GitHub.

