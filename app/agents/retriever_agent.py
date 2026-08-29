"""
Retriever agent.
Fetches the most relevant chunks from the knowledge base based on
the plan produced by the Planner agent.
"""

from app.services.vector_store import get_vector_store


def retrieve(plan: dict) -> dict:
    vector_store = get_vector_store()
    hits = vector_store.similarity_search(plan["question"], top_k=plan["top_k"])

    plan["retrieved_chunks"] = hits
    return plan
