# app.py
"""
"Get to Know Me" — Portfolio Chatbot for Sanket Muchhala
A Streamlit app that lets visitors have a conversation with Sanket's resume.

Run:
    streamlit run app.py

Required environment variables (set in .env or shell):
    GOOGLE_API_KEY  — Google Generative AI key for Gemini 1.5 Pro
                      Get one free at: https://aistudio.google.com/app/apikey
"""

import os
import sys
from pathlib import Path

# ── Environment setup (printed to terminal at startup) ────────────────────────
print("=" * 60)
print("  Get to Know Me — Portfolio Chatbot for Sanket Muchhala")
print("=" * 60)
print()
print("Required environment variables:")
print("  GOOGLE_API_KEY  — Gemini 1.5 Pro API key")
print("                    Get free key: https://aistudio.google.com/app/apikey")
print("                    Set in .env file:  GOOGLE_API_KEY=your_key_here")
print()
print("App will open at: http://localhost:8501")
print("=" * 60)
print()

from dotenv import load_dotenv
import streamlit as st

# Load .env for local development (no-op if running in cloud with env vars set)
load_dotenv()

# ── Page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Chat with Sanket | AI/ML Engineer",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] a {
        color: #64b5f6 !important;
    }

    /* Skill badge */
    .skill-badge {
        display: inline-block;
        background: rgba(100, 181, 246, 0.15);
        border: 1px solid rgba(100, 181, 246, 0.3);
        color: #90caf9;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        margin: 3px 2px;
    }

    /* Welcome header */
    .welcome-header {
        background: linear-gradient(135deg, #1565c0 0%, #6a1b9a 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .welcome-header h1 {
        margin: 0 0 0.4rem 0;
        font-size: 1.6rem;
    }
    .welcome-header p {
        margin: 0;
        opacity: 0.88;
        font-size: 0.95rem;
    }

    /* Suggestion button row */
    div[data-testid="stHorizontalBlock"] .stButton button {
        background: #f8f9ff;
        border: 1px solid #e3e8ff;
        border-radius: 10px;
        font-size: 0.85rem;
        color: #3c4a6e;
        padding: 0.5rem 0.8rem;
        text-align: left;
        width: 100%;
        transition: background 0.2s;
    }
    div[data-testid="stHorizontalBlock"] .stButton button:hover {
        background: #e8edff;
        border-color: #7b8ae0;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Sanket Muchhala")
    st.markdown("**AI/ML Engineer** · 3+ years")
    st.markdown("---")

    st.markdown("**🏢 Current Role**")
    st.markdown("AI Engineer @ Progressive Insurance")
    st.markdown("*(May 2024 – Present)*")

    st.markdown("**🎓 Education**")
    st.markdown("MS Data Science — Indiana University")
    st.markdown("B.Tech IT — Thakur College of Engineering")

    st.markdown("**⚡ Skills**")
    skills = [
        "Python", "LangChain", "RAG", "Agentic AI",
        "TensorFlow", "PyTorch", "AWS", "Azure",
        "ChromaDB", "FastAPI", "MLflow", "Apache Airflow",
    ]
    st.markdown(
        " ".join(f'<span class="skill-badge">{s}</span>' for s in skills),
        unsafe_allow_html=True,
    )

    st.markdown("**🏅 Certifications**")
    st.markdown(
        "AWS ML Engineer · Azure AI-900 · Databricks GenAI"
    )

    st.markdown("---")
    st.markdown("**🔗 Connect**")
    st.markdown(
        "[LinkedIn](https://linkedin.com/in/sanketmuchhala) &nbsp;|&nbsp; "
        "[GitHub](https://github.com/sanketmuchhala) &nbsp;|&nbsp; "
        "[Website](https://sanketmuchhala.com)",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.markdown(
        "<div style='font-size:0.72rem; opacity:0.5; margin-top:1rem;'>"
        "Powered by Gemini 1.5 Pro + ChromaDB + LangChain"
        "</div>",
        unsafe_allow_html=True,
    )


# ── API key guard ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ **GOOGLE_API_KEY not configured.**")
    st.info(
        "**To run this app:**\n\n"
        "1. Get a free API key at https://aistudio.google.com/app/apikey\n"
        "2. Create a `.env` file in this directory:\n"
        "   ```\n"
        "   GOOGLE_API_KEY=your_key_here\n"
        "   ```\n"
        "3. Restart the app:\n"
        "   ```bash\n"
        "   streamlit run app.py\n"
        "   ```"
    )
    st.stop()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None


# ── Load chain (once per session, not per message) ───────────────────────────
if st.session_state.chain is None:
    # Auto-build vector store if chromadb_data/ is missing (first-run or fresh clone)
    if not Path("chromadb_data").exists():
        with st.spinner("🔨 First-run setup: building knowledge base from resume…"):
            try:
                from backend.vector_store import build_vector_store
                build_vector_store()
            except Exception as e:
                st.error(f"Failed to build vector store: {e}")
                st.stop()

    with st.spinner("🧠 Loading Sanket's resume into memory…"):
        try:
            from backend.chat_engine import get_chat_engine
            st.session_state.chain = get_chat_engine(google_api_key=GOOGLE_API_KEY)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info(
                "Rebuild the vector store manually:\n"
                "```bash\npython3 -m backend.vector_store\n```"
            )
            st.stop()
        except EnvironmentError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize chat engine: {e}")
            st.stop()


# ── Main chat UI ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="welcome-header">
    <h1>💬 Ask me anything about Sanket</h1>
    <p>I'm an AI trained on Sanket's resume. Ask about his experience,
    technical skills, projects, or education.</p>
</div>
""", unsafe_allow_html=True)

# Suggested starter questions (only shown when conversation is fresh)
if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    suggestions = [
        "Where does Sanket currently work?",
        "What AI/ML skills does he have?",
        "Tell me about his notable projects",
        "What certifications does Sanket hold?",
    ]
    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        if col.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()

    st.markdown("")  # spacer

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask about Sanket's background, skills, or experience…"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.chain.invoke({"question": prompt})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                st.markdown(answer)

                # Collapsible source context (helpful for transparency)
                if source_docs:
                    with st.expander("📄 Resume context used", expanded=False):
                        for i, doc in enumerate(source_docs, 1):
                            page = doc.metadata.get("page", 0)
                            st.markdown(f"**Chunk {i}** *(page {page + 1})*")
                            st.text(doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""))
                            if i < len(source_docs):
                                st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                error_msg = f"Something went wrong: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )