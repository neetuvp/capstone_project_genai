"""
Response agent.
Takes the Reasoning agent's draft answer, runs it through the
guardrails/validation layer, and packages the final response with
sources — this is the "Result Verification agent" role from the
class notes (Task 9 folded into the agent pipeline).
"""

from app.core.guardrails import validate_answer


def respond(plan: dict) -> dict:
    validation = validate_answer(plan["draft_answer"], plan["retrieved_chunks"])

    sources = sorted({
        h["metadata"].get("source", "unknown")
        for h in plan["retrieved_chunks"]
    })

    return {
        "question": plan["question"],
        "answer": plan["draft_answer"],
        "sources": sources,
        "num_chunks_used": len(plan["retrieved_chunks"]),
        "strategy": plan["strategy"],
        "validation": validation,
    }
