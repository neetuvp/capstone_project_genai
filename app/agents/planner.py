"""
Planner agent.
Decides *how* to answer the question before any retrieval happens.
For this capstone, the planning is intentionally simple: it checks
whether the question is a simple lookup or a broader/comparative
question, and adjusts how many chunks to retrieve accordingly.
This is where you'd add multi-step decomposition for more complex
agents later.
"""

BROAD_KEYWORDS = ["compare", "summarize", "overview", "all", "list every", "across"]


def plan(question: str) -> dict:
    q_lower = question.lower()
    is_broad = any(kw in q_lower for kw in BROAD_KEYWORDS)

    return {
        "question": question,
        "top_k": 8 if is_broad else 4,
        "strategy": "broad_summary" if is_broad else "targeted_lookup",
    }
