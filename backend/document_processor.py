# backend/document_processor.py
"""
Loads the resume PDF and splits it into overlapping chunks
for downstream embedding and retrieval.

Uses pypdf directly (avoids deprecated langchain-community loader)
and produces LangChain Document objects compatible with all chains.
"""

from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to THIS file so the path is correct regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent.parent
RESUME_DIR = _PROJECT_ROOT / "resume"
RESUME_FILENAME = "Sanket_Muchhala_Resume.pdf"


# ── Core functions ────────────────────────────────────────────────────────────

def load_resume(resume_path: str = None) -> List[Document]:
    """
    Parse the resume PDF into a list of LangChain Document objects (one per page).

    Args:
        resume_path: Optional explicit path to the PDF.
                     Defaults to resume/Sanket_Muchhala_Resume.pdf in the project root.

    Returns:
        List of Document objects, each with page_content (str) and
        metadata containing 'source' (file path) and 'page' (0-indexed page number).

    Raises:
        FileNotFoundError: If the PDF cannot be found.
        RuntimeError: If the PDF contains no extractable text (e.g. image-only scan).
    """
    resolved = Path(resume_path) if resume_path else RESUME_DIR / RESUME_FILENAME

    if not resolved.exists():
        raise FileNotFoundError(
            f"Resume PDF not found at: {resolved}\n"
            f"Expected location: {RESUME_DIR / RESUME_FILENAME}\n"
            "Make sure the resume PDF is in the resume/ folder."
        )

    reader = PdfReader(str(resolved))

    pages: List[Document] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": str(resolved), "page": page_num},
                )
            )

    if not pages:
        raise RuntimeError(
            f"Could not extract any text from '{resolved}'.\n"
            "The PDF may be image-only (non-OCR). "
            "Consider running it through an OCR tool first."
        )

    return pages


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks for embedding.

    Args:
        documents: Raw Document objects from load_resume().
        chunk_size:    Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks
                       (preserves context across boundaries).

    Returns:
        List of chunked Document objects, each retaining source metadata.

    Raises:
        RuntimeError: If the splitter produces zero chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError(
            "RecursiveCharacterTextSplitter produced 0 chunks. "
            "The document may have no extractable text."
        )

    return chunks


def load_and_chunk_resume(resume_path: str = None) -> List[Document]:
    """
    Convenience function: load PDF then split into chunks.

    Args:
        resume_path: Optional path override (see load_resume).

    Returns:
        List of chunked Document objects ready for embedding.
    """
    pages = load_resume(resume_path)
    chunks = chunk_documents(pages)
    print(
        f"[document_processor] Loaded {len(pages)} page(s), "
        f"produced {len(chunks)} chunks."
    )
    return chunks


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running document_processor smoke test...")
    chunks = load_and_chunk_resume()
    print(f"\nFirst 3 chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n{'─'*60}")
        print(f"Chunk {i + 1} | page: {chunk.metadata.get('page', '?')}")
        print(chunk.page_content[:300])
    print(f"\n✓ Document processor working correctly ({len(chunks)} total chunks)")
