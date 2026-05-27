# backend/chat_engine.py
"""
Chat engine using a direct LCEL chain — one LLM call per question.

ConversationalRetrievalChain made two calls (condense + answer), doubling
token usage and hitting free-tier rate limits twice as fast. This version:
  - Retrieves relevant chunks from ChromaDB directly
  - Passes context + history into a single ChatPromptTemplate
  - Makes exactly ONE Gemini call per question
  - Fails fast (max_retries=1) instead of retrying for 5 minutes
"""

import os
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
for _log in ("langchain", "langchain_core", "langchain_classic"):
    logging.getLogger(_log).setLevel(logging.ERROR)

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from backend.vector_store import load_vector_store


# ── Persona ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an AI assistant representing Sanket Muchhala, an AI & Machine Learning Engineer. "
    "Answer questions about Sanket's experience, technical skills (RAG, Agentic AI, LLMs), "
    "and background using ONLY the resume context provided below. "
    "Be concise and professional. "
    "If the answer is not in the resume context, say so briefly and invite the visitor to "
    "connect with Sanket on LinkedIn (linkedin.com/in/sanketmuchhala).\n\n"
    "Resume context:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── Engine class ──────────────────────────────────────────────────────────────

class ChatEngine:
    """
    Thin wrapper around a retriever + LCEL chain.

    Usage:
        engine = ChatEngine(api_key)
        result = engine.invoke("Where does Sanket work?", history=[...])
        print(result["answer"])
        print(result["source_documents"])
    """

    def __init__(self, api_key: str):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.2,
            max_retries=1,        # fail fast — don't retry for 5 minutes
            convert_system_message_to_human=True,
        )

        vectorstore = load_vector_store()
        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        self._chain = _PROMPT | llm | StrOutputParser()

    def invoke(self, question: str, history: list = None) -> dict:
        """
        Args:
            question: The current user question.
            history:  List of prior {"role": "user"/"assistant", "content": str} dicts.

        Returns:
            {"answer": str, "source_documents": list[Document]}
        """
        # Convert history dicts → LangChain message objects
        lc_history = []
        for msg in (history or []):
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))

        # Retrieve relevant resume chunks
        source_docs = self._retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        # Single LLM call
        answer = self._chain.invoke({
            "question": question,
            "chat_history": lc_history,
            "context": context,
        })

        return {"answer": answer, "source_documents": source_docs}


# ── Factory ───────────────────────────────────────────────────────────────────

def get_chat_engine(google_api_key: Optional[str] = None) -> ChatEngine:
    """
    Build and return a ChatEngine instance.

    Args:
        google_api_key: If None, reads GOOGLE_API_KEY from the environment.

    Raises:
        EnvironmentError: No API key found.
        FileNotFoundError: ChromaDB not built yet — run python3 -m backend.vector_store
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No Google API key found.\n"
            "Set GOOGLE_API_KEY in your .env file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return ChatEngine(api_key)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Initializing chat engine...")
    engine = get_chat_engine()
    print("Ready.\n")

    q1 = "Where does Sanket currently work?"
    r1 = engine.invoke(q1)
    print(f"Q: {q1}")
    print(f"A: {r1['answer']}\n")

    q2 = "What did he do there?"
    r2 = engine.invoke(q2, history=[
        {"role": "user",      "content": q1},
        {"role": "assistant", "content": r1["answer"]},
    ])
    print(f"Q: {q2}")
    print(f"A: {r2['answer']}\n")
    print(f"Sources: {len(r2['source_documents'])} chunks")