# app.py
"""
"Get to Know Me" — Portfolio Chatbot for Sanket Muchhala
Run: streamlit run app.py
Requires: GOOGLE_API_KEY in .env  (https://aistudio.google.com/app/apikey)
"""

import os
import warnings
import logging
from pathlib import Path

# ── Silence noisy third-party warnings before any imports ────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

for _noisy_logger in (
    "sentence_transformers", "transformers", "chromadb",
    "langchain", "langchain_core", "langchain_classic",
    "httpx", "urllib3", "torch",
):
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

# ── Terminal startup info ─────────────────────────────────────────────────────
print("=" * 55)
print("  Get to Know Me  |  Sanket Muchhala Portfolio Bot")
print("=" * 55)
print("  GOOGLE_API_KEY  — required (Gemini 1.5 Pro)")
print("  Get one free:   https://aistudio.google.com/app/apikey")
print("  Set in .env:    GOOGLE_API_KEY=your_key_here")
print("=" * 55)
print()

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sanket Muchhala — AI/ML Engineer",
    page_icon="S",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #c9d1d9 !important;
    }
    section[data-testid="stSidebar"] a {
        color: #58a6ff !important;
        text-decoration: none;
    }
    section[data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }

    /* ── Name block ── */
    .profile-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.3px;
        margin: 0;
    }
    .profile-title {
        font-size: 0.82rem;
        color: #8b949e;
        margin: 2px 0 0 0;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    /* ── Sidebar section label ── */
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #484f58 !important;
        margin: 16px 0 4px 0;
    }

    /* ── Skill pill ── */
    .skill-pill {
        display: inline-block;
        background: #161b22;
        border: 1px solid #30363d;
        color: #8b949e;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        margin: 2px 2px 2px 0;
        font-family: 'Inter', monospace;
    }

    /* ── Hero banner ── */
    .hero {
        background: #0f1117;
        border: 1px solid #1e2130;
        border-radius: 12px;
        padding: 2rem 2.2rem 1.8rem;
        margin-bottom: 1.5rem;
    }
    .hero-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 0 0 4px 0;
    }
    .hero-sub {
        font-size: 0.9rem;
        color: #8b949e;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Suggestion buttons ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #0f1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        font-size: 0.82rem;
        color: #8b949e;
        padding: 0.55rem 0.9rem;
        text-align: left;
        width: 100%;
        transition: border-color 0.15s, color 0.15s;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: #58a6ff;
        color: #c9d1d9;
        background: #161b22;
    }

    /* ── Clear button ── */
    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid #30363d;
        color: #8b949e;
        border-radius: 6px;
        font-size: 0.8rem;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 10px;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-size: 0.82rem;
        color: #8b949e;
    }

    /* ── Footer ── */
    .sidebar-footer {
        font-size: 0.68rem;
        color: #484f58;
        margin-top: 1.5rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p class="profile-name">Sanket Muchhala</p>'
        '<p class="profile-title">AI / ML Engineer</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-label">Current Role</p>', unsafe_allow_html=True)
    st.markdown("AI Engineer @ Progressive Insurance  \n*(May 2024 – Present)*")

    st.markdown('<p class="sidebar-label">Education</p>', unsafe_allow_html=True)
    st.markdown(
        "MS Data Science — Indiana University  \n"
        "B.Tech IT — Thakur College of Engineering"
    )

    st.markdown('<p class="sidebar-label">Skills</p>', unsafe_allow_html=True)
    skills = [
        "Python", "LangChain", "RAG", "Agentic AI",
        "TensorFlow", "PyTorch", "AWS", "Azure",
        "ChromaDB", "FastAPI", "MLflow", "Airflow",
    ]
    st.markdown(
        "".join(f'<span class="skill-pill">{s}</span>' for s in skills),
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-label">Certifications</p>', unsafe_allow_html=True)
    st.markdown("AWS ML Engineer · Azure AI-900 · Databricks GenAI")

    st.markdown('<p class="sidebar-label">Connect</p>', unsafe_allow_html=True)
    st.markdown(
        "[LinkedIn](https://linkedin.com/in/sanketmuchhala) &nbsp;·&nbsp; "
        "[GitHub](https://github.com/sanketmuchhala) &nbsp;·&nbsp; "
        "[Website](https://sanketmuchhala.com)",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.markdown(
        '<p class="sidebar-footer">Gemini 2.0 Flash &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; LangChain</p>',
        unsafe_allow_html=True,
    )


# ── API key guard ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("**GOOGLE_API_KEY not configured.** Add it to your `.env` file and restart.")
    st.code("GOOGLE_API_KEY=your_key_here", language="bash")
    st.caption("Get a free key at https://aistudio.google.com/app/apikey")
    st.stop()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None


# ── Load chain once per session ───────────────────────────────────────────────
if st.session_state.chain is None:
    if not Path("chromadb_data").exists():
        with st.spinner("Building knowledge base from resume (first run)..."):
            try:
                from backend.vector_store import build_vector_store
                build_vector_store()
            except Exception as e:
                st.error(f"Failed to build vector store: {e}")
                st.stop()

    with st.spinner("Loading..."):
        try:
            from backend.chat_engine import get_chat_engine
            st.session_state.chain = get_chat_engine(google_api_key=GOOGLE_API_KEY)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except EnvironmentError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize chat engine: {e}")
            st.stop()


# ── Response helper ───────────────────────────────────────────────────────────
def _respond(question: str) -> None:
    """Call the chain, display the answer, and append it to session state."""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain.invoke({"question": question})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                st.markdown(answer)

                if source_docs:
                    with st.expander("Resume context used", expanded=False):
                        for i, doc in enumerate(source_docs, 1):
                            page = doc.metadata.get("page", 0)
                            st.caption(f"Chunk {i}  ·  page {page + 1}")
                            st.text(
                                doc.page_content[:300]
                                + ("..." if len(doc.page_content) > 300 else "")
                            )
                            if i < len(source_docs):
                                st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    msg = "Rate limit reached — please wait a moment and try again."
                elif "404" in err or "NOT_FOUND" in err:
                    msg = "Model not found. Check your API key or contact support."
                elif "403" in err or "PERMISSION_DENIED" in err:
                    msg = "API key does not have permission. Check your Google AI Studio settings."
                else:
                    msg = f"Something went wrong: {exc}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-name">Ask me about Sanket</p>
    <p class="hero-sub">
        I'm an AI trained on Sanket's resume. Ask about his work experience,
        technical skills, projects, or education.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Suggestion buttons (only when conversation is empty) ──────────────────────
if not st.session_state.messages:
    col1, col2 = st.columns(2)
    suggestions = [
        "Where does Sanket currently work?",
        "What AI/ML skills does he have?",
        "Tell me about his notable projects",
        "What certifications does he hold?",
    ]
    for i, suggestion in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        if col.button(suggestion, key=f"s{i}", use_container_width=True):
            # Store the question and rerun — _respond() will pick it up below
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()
    st.write("")


# ── Render all stored messages ────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Auto-respond if last message is an unanswered user question ───────────────
# This handles suggestion button clicks (which rerun without triggering chat_input)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    _respond(st.session_state.messages[-1]["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Sanket's background, skills, or experience..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    _respond(prompt)