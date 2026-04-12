🚀 RAG Document Chat App (Django + OpenAI)

A Retrieval-Augmented Generation (RAG) web app that allows users to upload and chat with documents (PDF, TXT, including scanned PDFs).

🔧 Tech Stack
Python / Django
OpenAI API (GPT-4)
FAISS (vector search)
PDF + text parsing

✨ Features
Upload and process PDF/TXT files
Supports scanned PDFs
Semantic search using embeddings
Chat interface for querying documents

🧠 How It Works
Documents are parsed and split into chunks
Embeddings are generated using OpenAI
FAISS indexes vectors for fast retrieval
GPT answers questions using retrieved context


git clone https://github.com/Kirito29/RAG.git
cd project

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

pip install -r requirements.txt

🔑 Environment Variables
Create a .env file: OPENAI_API_KEY=your_api_key_here

▶️ Run to build index
vector_store.build_index()

▶️ Run the App
python manage.py runserver

📂 Adding Your Documents
Place your files in:AI_Data/

