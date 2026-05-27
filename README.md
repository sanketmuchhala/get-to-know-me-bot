# 🤖 Get to Know Me — Portfolio Chatbot

An AI-powered chatbot that lets recruiters and collaborators ask natural-language questions about [Sanket Muchhala's](https://sanketmuchhala.com) background, skills, and experience.

Built with Python, LangChain, ChromaDB, Gemini 1.5 Pro, and Streamlit.

---

## ✨ Features

- **Natural conversation** — multi-turn chat with memory of prior messages
- **Resume-grounded answers** — responses are based solely on the resume (no hallucination)
- **Free embeddings** — `all-MiniLM-L6-v2` runs locally with no API cost
- **Transparent retrieval** — collapsible "Resume context used" expander shows which chunks were retrieved
- **Auto-setup** — first run auto-builds the vector store if `chromadb_data/` is missing

---

## 🏗️ Architecture

```
resume/Sanket_Muchhala_Resume.pdf
    │
    ├─ pypdf (text extraction)
    ├─ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    │       → 16 chunks
    │
    ├─ HuggingFaceEmbeddings (all-MiniLM-L6-v2, local)
    ├─ ChromaDB (persisted at ./chromadb_data/)
    │
    ├─ ConversationalRetrievalChain (langchain-classic)
    │       ├─ Retriever: top-3 similarity chunks
    │       ├─ Memory: ConversationBufferMemory
    │       └─ LLM: Gemini 1.5 Pro (langchain-google-genai)
    │
    └─ Streamlit chat UI (app.py)
```

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/sanketmuchhala/get-to-know-me-bot.git
cd get-to-know-me-bot
```

### 2. Set up virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90MB) on first use.
> It is cached locally and not re-downloaded on subsequent runs.

### 3. Configure your API key

```bash
cp .env.example .env
# Edit .env and paste your Google API key
```

Get a **free** Gemini API key at: https://aistudio.google.com/app/apikey

### 4. (Optional) Pre-build the vector store

The app auto-builds `chromadb_data/` on first run, but you can also build it manually:

```bash
python3 -m backend.vector_store
```

### 5. Run the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Google Generative AI API key (Gemini 1.5 Pro) |

---

## 📁 Project Structure

```
get-to-know-me-bot/
├── app.py                        # Streamlit frontend
├── backend/
│   ├── __init__.py
│   ├── document_processor.py    # PDF loading + text chunking
│   ├── vector_store.py          # Embeddings + ChromaDB
│   └── chat_engine.py           # LangChain chain + Gemini 1.5 Pro
├── resume/
│   └── Sanket_Muchhala_Resume.pdf
├── chromadb_data/                # Auto-generated, NOT in git
├── .env.example                  # Copy to .env and fill in values
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 1.5 Pro (`langchain-google-genai`) |
| Embeddings | `all-MiniLM-L6-v2` (`langchain-huggingface`) |
| Vector DB | ChromaDB (`langchain-chroma`) |
| Chain & Memory | LangChain Classic (`langchain-classic`) |
| PDF Parsing | pypdf + `langchain-text-splitters` |
| Frontend | Streamlit |

---

## 💬 Sample Questions

- *"Where does Sanket currently work?"*
- *"What AI/ML skills does he have?"*
- *"Tell me about his projects — LexOrchestrator, GTM Simulator"*
- *"What certifications does Sanket hold?"*
- *"Describe his experience at IBM and Indiana University"*

---

## 📜 License

MIT — feel free to fork and adapt for your own portfolio.