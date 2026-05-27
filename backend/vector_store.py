# backend/vector_store.py
"""
Embeds document chunks using sentence-transformers (free, local) and
stores them in a persistent ChromaDB collection at ./chromadb_data/.

Run standalone to build the database:
    python3 -m backend.vector_store
"""

import os
import warnings
import logging

# Silence noisy optional-dependency warnings (torchvision, torchaudio, etc.)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
for _log in ("sentence_transformers", "transformers", "chromadb", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)

from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.document_processor import load_and_chunk_resume


# ── Configuration ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PERSIST_DIR = str(_PROJECT_ROOT / "chromadb_data")
COLLECTION_NAME = "sanket_resume"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Embedding factory ─────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance using all-MiniLM-L6-v2.
    The model (~90MB) is cached locally by sentence-transformers on first run
    in ~/.cache/torch/sentence_transformers/.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Build ─────────────────────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document] = None) -> Chroma:
    """
    Embed document chunks and persist them to ChromaDB.

    If the database already exists, it is deleted and rebuilt from scratch
    so the store always reflects the current resume content.

    Args:
        chunks: Pre-loaded Document chunks. If None, loads from the resume PDF
                using load_and_chunk_resume().

    Returns:
        Chroma vectorstore instance pointing to CHROMA_PERSIST_DIR.

    Raises:
        RuntimeError: If embedding or persistence fails.
    """
    import shutil

    if chunks is None:
        chunks = load_and_chunk_resume()

    # Clean existing DB so we don't accumulate stale vectors on re-runs
    db_path = Path(CHROMA_PERSIST_DIR)
    if db_path.exists():
        print(f"[vector_store] Removing existing DB at {CHROMA_PERSIST_DIR}...")
        shutil.rmtree(db_path)

    print(
        f"[vector_store] Embedding {len(chunks)} chunks "
        f"with '{EMBEDDING_MODEL}' (first run downloads ~90MB)..."
    )

    embeddings = get_embeddings()

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build ChromaDB vector store: {exc}"
        ) from exc

    count = vectorstore._collection.count()
    print(f"[vector_store] ✓ ChromaDB ready — {count} vectors stored at {CHROMA_PERSIST_DIR}")
    return vectorstore


# ── Load ──────────────────────────────────────────────────────────────────────

def load_vector_store() -> Chroma:
    """
    Load an already-persisted ChromaDB collection from disk.

    Returns:
        Chroma vectorstore instance ready for similarity search / retrieval.

    Raises:
        FileNotFoundError: If chromadb_data/ does not exist yet.
                           Run `python3 -m backend.vector_store` first.
    """
    db_path = Path(CHROMA_PERSIST_DIR)
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{CHROMA_PERSIST_DIR}'.\n"
            "Build the vector store first by running:\n"
            "    python3 -m backend.vector_store"
        )

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building vector store from resume...")
    vs = build_vector_store()

    print("\nRunning similarity search test...")
    results = vs.similarity_search("AI Engineer skills and experience", k=3)
    print(f"\nTop {len(results)} results for 'AI Engineer skills and experience':")
    for i, doc in enumerate(results, 1):
        print(f"\n{'─'*60}")
        print(f"Result {i} | page: {doc.metadata.get('page', '?')}")
        print(doc.page_content[:200])

    print("\n✓ Vector store working correctly")
