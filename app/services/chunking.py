"""
Chunking service.
Splits long document text into overlapping chunks small enough
to embed meaningfully, while trying to break on paragraph/sentence
boundaries rather than mid-word (better retrieval quality than a
pure fixed-size splitter).
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Returns a list of dicts: {"text": chunk, "metadata": {...}}
    Metadata travels with each chunk into the vector store so we can
    later tell the user which document/chunk an answer came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph, then line, then sentence
    )
    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            "text": chunk,
            "metadata": {
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            },
        })
    return chunks
