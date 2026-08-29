"""
RAG pipeline.
The core "retrieve then generate" logic: take a user question,
pull the most relevant chunks from the vector store, and ask the
local LLM (via Ollama) to answer using ONLY that context.
This is what keeps answers grounded instead of hallucinated.
"""

from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from app.services.vector_store import get_vector_store

RAG_PROMPT_TEMPLATE = """You are an assistant that answers questions using ONLY the provided context.
If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer that."
Do not make up information that isn't in the context.

Context:
{context}

Question: {question}

Answer:"""


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a single context block, citing sources."""
    if not hits:
        return "(no relevant context found)"
    parts = []
    for h in hits:
        source = h["metadata"].get("source", "unknown")
        parts.append(f"[Source: {source}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def run_rag_query(question: str, top_k: int = None) -> dict:
    """
    Full RAG flow: embed query -> similarity search -> build prompt -> call LLM.
    Returns the answer plus the sources used, so the UI can show provenance.
    """
    vector_store = get_vector_store()
    hits = vector_store.similarity_search(question, top_k=top_k) if top_k else vector_store.similarity_search(question)

    context = build_context(hits)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    response = llm.invoke(prompt)

    sources = sorted({h["metadata"].get("source", "unknown") for h in hits})

    return {
        "answer": response.content,
        "sources": sources,
        "num_chunks_used": len(hits),
    }
