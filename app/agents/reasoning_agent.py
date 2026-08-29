"""
Reasoning agent.
Analyses the retrieved context and generates a draft answer using
the local LLM (via Ollama). This is the "thinking" step, separate
from the final Response agent which packages and validates output.
"""

from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL

REASONING_PROMPT = """You are an assistant that answers questions using ONLY the provided context.
If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer that."
Do not make up information that isn't in the context.

Strategy: {strategy}

Context:
{context}

Question: {question}

Answer:"""


def _build_context(hits: list[dict]) -> str:
    if not hits:
        return "(no relevant context found)"
    parts = []
    for h in hits:
        source = h["metadata"].get("source", "unknown")
        parts.append(f"[Source: {source}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def reason(plan: dict) -> dict:
    context = _build_context(plan["retrieved_chunks"])
    prompt = REASONING_PROMPT.format(
        strategy=plan["strategy"],
        context=context,
        question=plan["question"],
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    response = llm.invoke(prompt)

    plan["draft_answer"] = response.content
    return plan
