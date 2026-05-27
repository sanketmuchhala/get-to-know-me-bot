# backend/chat_engine.py
"""
Configures the ConversationalRetrievalChain with:
  - Gemini 1.5 Pro as the LLM (via langchain-google-genai)
  - all-MiniLM-L6-v2 embeddings — free, local, no API cost
  - ChromaDB as the persistent vector store
  - ConversationBufferMemory for natural multi-turn conversations

Note: In LangChain 1.x, ConversationalRetrievalChain and
ConversationBufferMemory live in the `langchain_classic` package.
"""

import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from backend.vector_store import load_vector_store


# ── System persona & prompt ───────────────────────────────────────────────────

_SYSTEM_CONTEXT = (
    "You are an AI assistant representing Sanket Muchhala, an AI & Machine Learning Engineer. "
    "Your job is to accurately and professionally answer questions about Sanket's experience, "
    "technical skills (including RAG, Agentic architectures, LLMs), and background using ONLY "
    "the provided resume context. "
    "Be concise, warm, and professional — like a knowledgeable colleague. "
    "If a question falls outside the resume context, politely explain you can only speak to "
    "what is on the resume and invite the visitor to connect with Sanket directly on LinkedIn "
    "(linkedin.com/in/sanketmuchhala)."
)

# This prompt is injected into the StuffDocumentsChain that combines
# retrieved resume chunks with the question.
_QA_PROMPT_TEMPLATE = f"""{_SYSTEM_CONTEXT}

Resume context:
{{context}}

Chat history:
{{chat_history}}

Visitor question: {{question}}

Answer:"""

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=_QA_PROMPT_TEMPLATE,
)


# ── Factory function ──────────────────────────────────────────────────────────

def get_chat_engine(google_api_key: Optional[str] = None) -> ConversationalRetrievalChain:
    """
    Build and return a ready-to-use ConversationalRetrievalChain.

    The chain:
    1. Reformulates follow-up questions into standalone queries (condense step)
    2. Retrieves the top-3 most relevant resume chunks from ChromaDB
    3. Passes chunks + conversation history to Gemini 1.5 Pro with the persona prompt
    4. Persists the exchange in memory for follow-up awareness

    Args:
        google_api_key: Google Generative AI API key. If None, reads from
                        the GOOGLE_API_KEY environment variable.

    Returns:
        Configured ConversationalRetrievalChain. Call with:
            result = chain.invoke({"question": "..."})
            answer = result["answer"]
            sources = result["source_documents"]

    Raises:
        EnvironmentError: If no Google API key is found.
        FileNotFoundError: If the ChromaDB vector store has not been built yet.
    """
    # ── 1. Resolve API key ────────────────────────────────────────────────────
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No Google API key found.\n"
            "Set GOOGLE_API_KEY in your .env file or pass it to get_chat_engine().\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    # ── 2. LLM: Gemini 1.5 Pro ───────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key,
        temperature=0.2,          # Low temperature → factual, professional answers
        convert_system_message_to_human=True,  # Required: Gemini rejects system role
    )

    # ── 3. Retriever: top-3 chunks from ChromaDB ─────────────────────────────
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # ── 4. Memory: tracks conversation turns ─────────────────────────────────
    # output_key="answer" prevents ambiguity when return_source_documents=True
    # return_messages=False keeps history as a plain string (required by PromptTemplate)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=False,
    )

    # ── 5. Build the chain ────────────────────────────────────────────────────
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )

    return chain


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick end-to-end test.
    Requires GOOGLE_API_KEY to be set in .env or environment.

    Run:
        python3 -m backend.chat_engine
    """
    from dotenv import load_dotenv

    load_dotenv()

    print("Initializing chat engine (loading embeddings + ChromaDB)...")
    chain = get_chat_engine()
    print("✓ Chat engine ready\n")

    test_questions = [
        "Where does Sanket currently work?",
        "What AI and machine learning skills does he have?",
        "Tell me about his education.",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = chain.invoke({"question": q})
        print(f"A: {result['answer'][:300]}")
        print(f"   (Sources: {len(result.get('source_documents', []))} chunks)")
        print()
